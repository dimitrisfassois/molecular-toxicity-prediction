"""Executes model training."""
import argparse
import logging
import fsspec
import os
import sys
from collections.abc import Sequence as SequenceCollection

import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import torch
import deepchem as dc
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.layers import GraphConv, GraphPool, GraphGather
from deepchem.metrics import to_one_hot
from deepchem.utils.evaluate import _process_metric_input


# sys.path.append('..')
# Get the absolute path of the current script file
current_file = os.path.abspath(__file__)

# Derive the parent directory from the current file's path
parent_dir = os.path.dirname(current_file)

# Add the parent directory to the Python module search path
sys.path.append(parent_dir)


# from util.constants import CONST
# from data_loaders.data_loaders_utils import get_generator
# from data_loaders.data_loaders import get_disk_dataset
# from models.callbacks import ValidationCallback
# from trainer.model_training import get_model

def constant(f):
    def fset(self, value):
        raise TypeError
    def fget(self):
        return f()
    return property(fget, fset)

class _Constants(object):
    @constant
    def TASKS():
        return ['NR-AR',
         'NR-AR-LBD',
         'NR-AhR',
         'NR-Aromatase',
         'NR-ER',
         'NR-ER-LBD',
         'NR-PPAR-gamma',
         'SR-ARE',
         'SR-ATAD5',
         'SR-HSE',
         'SR-MMP',
         'SR-p53']

CONST = _Constants()

def full_dataset_generator(dataset, batch_size, n_epochs, n_tasks, deterministic):
    for ind, (X_b, y_b, w_b, ids_b) in enumerate(
        dataset.iterbatches(batch_size, n_epochs, deterministic=deterministic, pad_batches=True)
    ):
        multiConvMol = ConvMol.agglomerate_mols(X_b[:, -1])
        graph_inputs = [
            multiConvMol.get_atom_features(),
            multiConvMol.deg_slice,
            np.array(multiConvMol.membership),
        ]
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
            graph_inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
        fc_net_inputs = [X_b[:, :-1].astype("float32")]
        labels = [to_one_hot(y_b.flatten(), 2).reshape(-1, n_tasks, 2)]
        weights = [w_b]
        yield fc_net_inputs + graph_inputs, labels, weights

def fingerprints_generator(dataset, batch_size, n_epochs, n_tasks, deterministic):
    for ind, (X_b, y_b, w_b, ids_b) in enumerate(
        dataset.iterbatches(batch_size, n_epochs, deterministic=deterministic, pad_batches=True)
    ):
        fc_net_inputs = [X_b[:, :1024].astype("float32")]
        labels = [to_one_hot(y_b.flatten(), 2).reshape(-1, n_tasks, 2)]
        weights = [w_b]
        yield fc_net_inputs, labels, weights


def get_generator(model_type, dataset, batch_size, n_epochs, n_tasks, deterministic=False):
    if model_type in ("self_attention", "multi_headed_attention"):
        generator = full_dataset_generator(dataset, batch_size, n_epochs, n_tasks, deterministic)
    elif model_type == "multi_task_classifier":
        generator = fingerprints_generator(dataset, batch_size, n_epochs, n_tasks, deterministic)
    return generator


def get_disk_dataset(fs, data_dir):
    return dc.data.datasets.DiskDataset.from_numpy(
        X=np.load(fs.open(f"{data_dir}/shard-0-X.npy"), allow_pickle=True),
        y=np.load(fs.open(f"{data_dir}/shard-0-y.npy"), allow_pickle=True),
        w=np.load(fs.open(f"{data_dir}/shard-0-w.npy"), allow_pickle=True),
        ids=np.load(fs.open(f"{data_dir}/shard-0-ids.npy"), allow_pickle=True),
        tasks=CONST.TASKS
    )


class ValidationCallback(object):
    """Performs validation while training a KerasModel.
    This is a callback that can be passed to fit().  It periodically computes a
    set of metrics over a validation set, writes them to a file, and keeps track
    of the best score. In addition, it can save the best model parameters found
    so far to a directory on disk, updating them every time it finds a new best
    validation score.
    If Tensorboard logging is enabled on the KerasModel, the metrics are also
    logged to Tensorboard.  This only happens when validation coincides with a
    step on which the model writes to the log.  You should therefore make sure
    that this callback's reporting interval is an even fraction or multiple of
    the model's logging interval.
    """

    def __init__(self,
                 dataset,
                 interval,
                 metrics,
                 n_tasks,
                 model_type,
                 batch_size,
                 output_file=sys.stdout,
                 save_dir=None,
                 save_metric=0,
                 save_on_minimum=False,
                 transformers=[]):
        """Create a ValidationCallback.
        Parameters
        ----------
        dataset: dc.data.Dataset
            the validation set on which to compute the metrics
        interval: int
            the interval (in training steps) at which to perform validation
        metrics: list of dc.metrics.Metric
            metrics to compute on the validation set
        output_file: file
            to file to which results should be written
        save_dir: str
            if not None, the model parameters that produce the best validation score
            will be written to this directory
        save_metric: int
            the index of the metric to use when deciding whether to write a new set
            of parameters to disk
        save_on_minimum: bool
            if True, the best model is considered to be the one that minimizes the
            validation metric.  If False, the best model is considered to be the one
            that maximizes it.
        transformers: List[Transformer]
            List of `dc.trans.Transformer` objects. These transformations
            must have been applied to `dataset` previously. The dataset will
            be untransformed for metric evaluation.
        """
        self.dataset = dataset
        self.interval = interval
        self.metrics = metrics
        self.output_file = output_file
        self.save_dir = save_dir
        self.save_metric = save_metric
        self.save_on_minimum = save_on_minimum
        self._best_score = None
        self.transformers = transformers
        self.n_tasks = n_tasks
        self.model_type = model_type
        self.batch_size = batch_size

    def __call__(self, model, step):
        """This is invoked by the KerasModel after every step of fitting.
        Parameters
        ----------
        model: KerasModel
            the model that is being trained
        step: int
            the index of the training step that has just completed
        """
        if step % self.interval != 0:
            return

        val_generator = get_generator(
            self.model_type,
            self.dataset,
            self.batch_size,
            1,
            self.n_tasks,
            deterministic=True,
        )
        metrics = _process_metric_input(self.metrics)

        y = self.dataset.y
        w = self.dataset.w
        y_pred = model.predict_on_generator(val_generator)
        y_pred = y_pred[:self.dataset.y.shape[0]]

        scores = {}
        per_task_metrics = False
        n_classes = 2
        use_sample_weights = True
        for metric in metrics:
            results = metric.compute_metric(
                y,
                y_pred,
                w,
                per_task_metrics=per_task_metrics,
                n_tasks=self.n_tasks,
                n_classes=n_classes,
                use_sample_weights=use_sample_weights)
            scores[metric.name] = results

        message = 'Step %d validation:' % step
        for key in scores:
            message += ' %s=%g' % (key, scores[key])
        print(message, file=self.output_file)
        if model.tensorboard:
            for key in scores:
                model._log_scalar_to_tensorboard(key, scores[key],
                                                 model.get_global_step())
        score = scores[metrics[self.save_metric].name]
        if not self.save_on_minimum:
            score = -score
        if self._best_score is None or score < self._best_score:
            self._best_score = score
            if self.save_dir is not None:
                model.save_checkpoint(model_dir=self.save_dir,  max_checkpoints_to_keep=30)
        if model.wandb_logger is not None:
            # Log data to Wandb
            data = {'eval/' + k: v for k, v in scores.items()}
            model.wandb_logger.log_data(data, step, dataset_id=id(self.dataset))

    def get_best_score(self):
        """This getter returns the best score evaluated on the validation set.
        Returns
        -------
        float
            The best score.
        """
        if self.save_on_minimum:
            return self._best_score
        else:
            return -self._best_score


def get_loss():
    """

    :return:
    """
    loss = dc.models.losses.CategoricalCrossEntropy()
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    return loss, metric


class SelfAttentionModel(tf.keras.Model):
    def __init__(self, n_tasks, n_features, batch_size):
        super(SelfAttentionModel, self).__init__(n_tasks, n_features, batch_size)
        self.n_tasks = n_tasks
        self.fcn = FCN(n_tasks=n_tasks, n_features=n_features, layer_sizes=[1000])
        self.graph_model = GraphConvModel(batch_size)
        self.attention_layer = tf.keras.layers.Attention(use_scale=True)
        self.output_layer = layers.Dense(n_tasks*2)
        self.logits = layers.Reshape((n_tasks, 2))
        self.softmax = layers.Softmax()

    def call(self, inputs):
        fcn_output = self.fcn(inputs[0])
        graph_model_output = self.graph_model(inputs[1:])
        concatenated = tf.concat([fcn_output, graph_model_output], axis=-1)
        attention = self.attention_layer([concatenated, concatenated])
        logits_output = self.logits(self.output_layer(attention))
        return self.softmax(logits_output)


class GraphConvModel(tf.keras.Model):
    def __init__(self, batch_size):
        super(GraphConvModel, self).__init__(batch_size)
        self.gc1 = GraphConv(128, activation_fn=tf.nn.tanh)
        self.batch_norm1 = layers.BatchNormalization()
        self.gp1 = GraphPool()

        self.gc2 = GraphConv(128, activation_fn=tf.nn.tanh)
        self.batch_norm2 = layers.BatchNormalization()
        self.gp2 = GraphPool()

        self.dense1 = layers.Dense(256, activation=tf.nn.tanh)
        self.batch_norm3 = layers.BatchNormalization()
        self.readout = GraphGather(batch_size=batch_size, activation_fn=tf.nn.tanh)

    def call(self, inputs):
        gc1_output = self.gc1(inputs)
        batch_norm1_output = self.batch_norm1(gc1_output)
        gp1_output = self.gp1([batch_norm1_output] + inputs[1:])

        gc2_output = self.gc2([gp1_output] + inputs[1:])
        batch_norm2_output = self.batch_norm1(gc2_output)
        gp2_output = self.gp2([batch_norm2_output] + inputs[1:])

        dense1_output = self.dense1(gp2_output)
        batch_norm3_output = self.batch_norm3(dense1_output)
        readout_output = self.readout([batch_norm3_output] + inputs[1:])

        return readout_output


class FCN(tf.keras.Model):
    def __init__(
        self,
        n_tasks,
        n_features,
        layer_sizes=[1000],
        weight_init_stddevs=0.02,
        bias_init_consts=1.0,
        weight_decay_penalty=0.0,
        weight_decay_penalty_type="l2",
        dropouts=0.5,
        activation_fns="relu",
        residual=False,
        **kwargs
    ) -> None:
        super(FCN, self).__init__()
        self.n_tasks = n_tasks
        self.n_features = n_features
        self.layer_sizes = layer_sizes
        n_layers = len(layer_sizes)
        if not isinstance(weight_init_stddevs, SequenceCollection):
            weight_init_stddevs = [weight_init_stddevs] * n_layers
        if not isinstance(bias_init_consts, SequenceCollection):
            bias_init_consts = [bias_init_consts] * n_layers
        if not isinstance(dropouts, SequenceCollection):
            dropouts = [dropouts] * n_layers
        if isinstance(activation_fns, str) or not isinstance(
            activation_fns, SequenceCollection
        ):
            activation_fns = [activation_fns] * n_layers
        activation_fns = [getattr(tf.keras.activations, f) for f in activation_fns]

        self._layers = []
        self.dropouts = dropouts
        self.activation_fns = activation_fns
        self.residual = residual
        prev_size = n_features
        for size, weight_stddev, bias_const in zip(
            layer_sizes, weight_init_stddevs, bias_init_consts
        ):
            layer = tf.keras.layers.Dense(
                size,
                kernel_initializer=tf.random_normal_initializer(stddev=weight_stddev),
                bias_initializer=tf.constant_initializer(bias_const),
            )
            self._layers.append(layer)
            prev_size = size

    def call(self, x, training=False):
        prev_size = self.n_features
        next_activation = None
        for size, layer, dropout, activation_fn in zip(
            self.layer_sizes, self._layers, self.dropouts, self.activation_fns
        ):
            y = x
            if next_activation is not None:
                y = next_activation(x)
            y = layer(y)
            if dropout > 0.0:
                y = tf.keras.layers.Dropout(dropout)(y)
            if self.residual and prev_size == size:
                y = x + y
            x = y
            prev_size = size
            next_activation = activation_fn
        if next_activation is not None:
            y = next_activation(y)
        neural_fingerprint = y
        return neural_fingerprint


class MultiHeadedAttentionModel(tf.keras.Model):
    def __init__(self, n_tasks, n_features, batch_size, num_heads, key_dim):
        super(MultiHeadedAttentionModel, self).__init__(n_tasks)
        self.n_tasks = n_tasks
        self.fcn = FCN(n_tasks=12, n_features=n_features, layer_sizes=[1000])
        self.graph_model = GraphConvModel(batch_size=batch_size)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=int(num_heads), key_dim=int(key_dim))
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.ffn = FCN(n_tasks=n_tasks, n_features=n_features)
        self.output_layer = layers.Dense(n_tasks*2)
        self.logits = layers.Reshape((n_tasks, 2))
        self.softmax = layers.Softmax()

    def call(self, inputs):
        fcn_output = self.fcn(inputs[0])
        graph_model_output = self.graph_model(inputs[1:])
        concatenated = tf.concat([fcn_output, graph_model_output], axis=-1)
        attn_output = self.mha(
            query=tf.expand_dims(concatenated, -1),
            value=tf.expand_dims(concatenated, -1),
            key=tf.expand_dims(concatenated, -1))
        attn_output = tf.squeeze(attn_output)
        concatenated = self.add([concatenated, attn_output])
        concatenated = self.layernorm(concatenated)
        concatenated = self.ffn(concatenated)
        logits_output = self.logits(self.output_layer(concatenated))
        return self.softmax(logits_output)


def get_model(model_type, n_tasks, batch_size, model_dir, learning_rate, num_heads, key_dim):
    """

    :param model_type:
    :param n_tasks:
    :param batch_size:
    :param model_dir:
    :param learning_rate:
    :return:
    """
    assert model_type in (
        "multi_headed_attention",
        "self_attention",
        "multi_task_classifier",
    ), "Invalid input for model type!"

    loss, metric = get_loss()
    if model_type == "multi_headed_attention":
        n_features = 1792
        model = MultiHeadedAttentionModel(n_tasks, n_features, batch_size, num_heads, key_dim)
        dc_model = dc.models.KerasModel(model=model, loss=loss, model_dir=model_dir, learning_rate=learning_rate)
    elif model_type == "self_attention":
        n_features = 1792
        model = SelfAttentionModel(n_tasks, n_features, batch_size)
        dc_model = dc.models.KerasModel(model=model, loss=loss, model_dir=model_dir, learning_rate=learning_rate)
    elif model_type == "multi_task_classifier":
        n_features = 1024
        dc_model = dc.models.MultitaskClassifier(
            n_tasks=n_tasks, n_features=n_features, layer_sizes=[1000], model_dir=model_dir, learning_rate=learning_rate
        )
    return dc_model


def _train_model():
    logging.info("Loading training, validation and test datasets.")
    project_id = "molecular-toxicity-prediction"
    fs = fsspec.filesystem("gs", project=project_id)
    train_data_dir = "gs://molecular-toxicity-prediction/data/train_dataset"
    val_data_dir = "gs://molecular-toxicity-prediction/data/val_dataset"
    train_dataset = get_disk_dataset(fs, train_data_dir)
    val_dataset = get_disk_dataset(fs, val_data_dir)

    model_type = "self_attention"
    logging.info(f"Initializing model: {model_type}")
    n_tasks = len(CONST.TASKS)
    learning_rate = dc.models.optimizers.ExponentialDecay(0.0002, 0.9, 1000)
    model_dir = "gs://molecular-toxicity-prediction/model_checkpoints/self_attention/kubeflow_pipeline/run1"
    batch_size = 50
    dc_model = get_model(
        model_type,
        n_tasks,
        batch_size,
        model_dir,
        learning_rate,
        0,
        0,
    )
    n_epochs = 2

    logging.info(f"Starting model training for {n_epochs} epochs.")
    train_generator = get_generator(
        model_type, train_dataset, batch_size, n_epochs, n_tasks
    )
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

    save_dir = "gs://molecular-toxicity-prediction/callback_checkpoints/self_attention/kubeflow_pipeline/run1"
    logging.info(f"Setting up early stopping in the {save_dir} directory")
    callback = ValidationCallback(
        val_dataset,
        100,
        metric,
        n_tasks,
        model_type,
        batch_size,
        save_dir=save_dir,
    )

    avg_loss = dc_model.fit_generator(train_generator, callbacks=callback)
    logging.info(f"Average loss over the most recent checkpoint interval: {avg_loss}")

    if model_type == "multi_task_classifier":
        data = {
            "model_state_dict": dc_model.model.state_dict(),
            "optimizer_state_dict": dc_model._pytorch_optimizer.state_dict(),
            "global_step": dc_model._global_step,
        }
        with fs.open(save_dir, mode="wb") as f:
            torch.save(data, f)


if __name__ == '__main__':
    _train_model()
