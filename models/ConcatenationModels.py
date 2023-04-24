import tensorflow as tf
import tensorflow.keras.layers as layers

from .GraphConvModel import GraphConvModel
from .FCN import FCN

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


class MultiHeadedAttentionModel(tf.keras.Model):
    def __init__(self, n_tasks, n_features, batch_size, num_heads, key_dim):
        super(MultiHeadedAttentionModel, self).__init__(n_tasks)
        self.n_tasks = n_tasks
        self.fcn = FCN(n_tasks=12, n_features=n_features, layer_sizes=[1000])
        self.graph_model = GraphConvModel(batch_size=batch_size)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
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
