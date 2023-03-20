"""
Callback functions that can be invoked while fitting a KerasModel.
This is an adaptation from the deepchem implementation to work with a validation generator:
https://github.com/deepchem/deepchem/blob/b03e2d3e378bc74765b539ead70ded4afdbaeaee/deepchem/models/callbacks.py
"""
import sys

from deepchem.utils.evaluate import _process_metric_input

from data_loaders.data_loaders_utils import get_generator


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
                model.save_checkpoint(model_dir=self.save_dir)
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