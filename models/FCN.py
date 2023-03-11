import tensorflow as tf
from deepchem.utils.typing import ActivationFn, OneOrMany
from collections.abc import Sequence as SequenceCollection


class FCN(tf.keras.Model):
    def __init__(
        self,
        n_tasks: int,
        n_features: int,
        layer_sizes: SequenceCollection[int] = [1000],
        weight_init_stddevs: OneOrMany[float] = 0.02,
        bias_init_consts: OneOrMany[float] = 1.0,
        weight_decay_penalty: float = 0.0,
        weight_decay_penalty_type: str = "l2",
        dropouts: OneOrMany[float] = 0.5,
        activation_fns: OneOrMany[ActivationFn] = "relu",
        residual: bool = False,
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
