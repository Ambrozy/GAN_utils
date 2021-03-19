import tensorflow as tf
import tensorflow.keras.models as M
import tensorflow.keras.layers as L


class TransformerBlock(L.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = L.MultiHeadAttention(num_heads, key_dim=embed_dim)
        self.ffn = M.Sequential([
            L.Dense(ff_dim, activation='relu'),
            L.Dense(embed_dim),
        ])
        self.layernorm1 = L.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = L.LayerNormalization(epsilon=1e-6)
        self.dropout1 = L.Dropout(rate)
        self.dropout2 = L.Dropout(rate)

    def call(self, input, **kwargs):
        x = input
        x = self.att(x, x)
        x = self.dropout1(x, training=kwargs['training'])
        x = input = self.layernorm1(input + x)
        x = self.ffn(x)
        x = self.dropout2(x, training=kwargs['training'])
        return self.layernorm2(input + x)


class Transformer2DLayer(L.Layer):
    def __init__(self, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        self.w = int(input_shape[-3])
        self.h = int(input_shape[-2])
        self.d = int(input_shape[-1])
        self.att = TransformerBlock(self.d, self.num_heads, self.d)

    def call(self, x, **kwargs):
        x = tf.reshape(x, (-1, self.w * self.h, self.d))
        x = self.att(x)
        x = tf.reshape(x, (-1, self.w, self.h, self.d))
        return x
