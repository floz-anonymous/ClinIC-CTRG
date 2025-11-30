import tensorflow as tf
from transformers import TFBertModel


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ]
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        attn1, attn_weights_block1 = self.mha1(
            x, x, x, attention_mask=look_ahead_mask, return_attention_scores=True
        )
        attn1 = self.dropout1(attn_output=attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, attn_weights_block2 = self.mha2(
            out1,
            enc_output,
            enc_output,
            attention_mask=padding_mask,
            return_attention_scores=True,
        )
        attn2 = self.dropout2(attn_output=attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3, attn_weights_block1, attn_weights_block2


class KnowledgeAugmentedDecoder(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        num_layers=2,
        num_heads=8,
        dff=2048,
        dropout_rate=0.1,
    ):
        super(KnowledgeAugmentedDecoder, self).__init__()

        self.knowledge_encoder = TFBertModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT", from_pt=True
        )
        self.knowledge_projector = tf.keras.layers.Dense(embedding_dim)

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = self.positional_encoding(2048, embedding_dim)

        self.dec_layers = [
            TransformerDecoderLayer(embedding_dim, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def positional_encoding(self, position, d_model):

        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model,
        )

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.float32(d_model))
        return pos * angle_rates

    def call(
        self,
        fused_multimodal_features,
        knowledge_input_ids,
        knowledge_attention_mask,
        target_report_ids,
        training=False,
    ):
        knowledge_outputs = self.knowledge_encoder(
            input_ids=knowledge_input_ids, attention_mask=knowledge_attention_mask
        )
        z_kg = knowledge_outputs.last_hidden_state
        z_kg = self.knowledge_projector(z_kg)

        if len(fused_multimodal_features.shape) == 2:
            fused_multimodal_features = tf.expand_dims(
                fused_multimodal_features, axis=1
            )

        combined_context = tf.concat([fused_multimodal_features, z_kg], axis=1)

        seq_len = tf.shape(target_report_ids)[1]

        x = self.embedding(target_report_ids)
        x *= tf.math.sqrt(tf.cast(self.embedding.output_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        look_ahead_mask = None
        padding_mask = None

        for i in range(len(self.dec_layers)):
            x, block1, block2 = self.dec_layers[i](
                x, combined_context, training, look_ahead_mask, padding_mask
            )

        logits = self.final_layer(x)

        return logits


def retrieve_knowledge(input_image_embedding, ct_repository_embeddings, k=3):

    input_norm = tf.math.l2_normalize(input_image_embedding, axis=-1)
    repo_norm = tf.math.l2_normalize(ct_repository_embeddings, axis=-1)

    sims = tf.matmul(input_norm, repo_norm, transpose_b=True)

    values, top_k_indices = tf.math.top_k(sims, k=k)

    retrieved_reports = [all_reports[i] for i in top_k_indices.numpy()[0]]

    return retrieved_reports
