import tensorflow as tf
from transformers import TFGitVisionModel, TFConvNextV2Model

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim_feedforward, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.self_attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class DualPathVisualEncoder(tf.keras.Model):
    def __init__(self, embedding_dim=512, num_slices=32, pretrained=True):
        super(DualPathVisualEncoder, self).__init__()
        
        self.strong_path_encoder = TFGitVisionModel.from_pretrained("microsoft/git-base")
        self.strong_projector = tf.keras.layers.Dense(embedding_dim)

        self.weak_path_encoder = TFConvNextV2Model.from_pretrained("facebook/convnext-v2-atto-1k-224")
        self.weak_projector = tf.keras.layers.Dense(embedding_dim)

        self.slice_positional_encoding = self.add_weight(
            name="slice_positional_encoding",
            shape=(1, num_slices, embedding_dim),
            initializer="random_normal",
            trainable=True
        )

        self.volumetric_context_transformer = [
            TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=2048)
            for _ in range(2)
        ]

    def call(self, input_ct_volume, training=False):
        batch_size = tf.shape(input_ct_volume)[0]
        num_slices = tf.shape(input_ct_volume)[1]
        channels = tf.shape(input_ct_volume)[2]
        height = tf.shape(input_ct_volume)[3]
        width = tf.shape(input_ct_volume)[4]

        reshaped_slices = tf.reshape(input_ct_volume, (batch_size * num_slices, channels, height, width))
        
        git_outputs = self.strong_path_encoder(pixel_values=reshaped_slices, training=training)
        git_features = tf.reduce_mean(git_outputs.last_hidden_state, axis=1)
        visual_features_strong = self.strong_projector(git_features)
        visual_features_strong = tf.reshape(visual_features_strong, (batch_size, num_slices, -1))

        convnext_outputs = self.weak_path_encoder(pixel_values=reshaped_slices, training=training)
        
        convnext_features = convnext_outputs.pooler_output
        visual_features_weak = self.weak_projector(convnext_features)
        visual_features_weak = tf.reshape(visual_features_weak, (batch_size, num_slices, -1))

        visual_features_strong = visual_features_strong + self.slice_positional_encoding[:, :num_slices, :]
        visual_features_weak = visual_features_weak + self.slice_positional_encoding[:, :num_slices, :]

        f_s_v_sequence = visual_features_strong
        for layer in self.volumetric_context_transformer:
            f_s_v_sequence = layer(f_s_v_sequence, training=training)

        f_w_v_sequence = visual_features_weak
        for layer in self.volumetric_context_transformer:
            f_w_v_sequence = layer(f_w_v_sequence, training=training)

        f_s_v = tf.reduce_mean(f_s_v_sequence, axis=1)
        f_w_v = tf.reduce_mean(f_w_v_sequence, axis=1)

        return f_s_v, f_w_v