import tensorflow as tf


class FusionBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(FusionBlock, self).__init__()
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.feed_forward_network = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, query_features, key_value_features, training=False):
        attention_output = self.multi_head_attention(
            query=query_features, value=key_value_features, key=key_value_features
        )
        attention_output = self.dropout_1(attention_output, training=training)

        normalized_attention_output = self.layer_norm_1(
            query_features + attention_output
        )

        ffn_output = self.feed_forward_network(normalized_attention_output)
        ffn_output = self.dropout_2(ffn_output, training=training)

        return self.layer_norm_2(normalized_attention_output + ffn_output)


class StrongWeakMultimodalFusion(tf.keras.Model):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=2048, dropout_rate=0.1):
        super(StrongWeakMultimodalFusion, self).__init__()

        self.visual_to_text_fusion_block = FusionBlock(
            embed_dim, num_heads, ff_dim, dropout_rate
        )

        self.text_to_visual_fusion_block = FusionBlock(
            embed_dim, num_heads, ff_dim, dropout_rate
        )

        self.concatenation_layer = tf.keras.layers.Concatenate(axis=-1)

    def call(
        self,
        strong_visual_embeddings,
        weak_visual_embeddings,
        strong_textual_embeddings,
        weak_textual_embeddings,
        training=False,
    ):
        if len(strong_visual_embeddings.shape) == 2:
            strong_visual_embeddings = tf.expand_dims(strong_visual_embeddings, axis=1)
        if len(weak_visual_embeddings.shape) == 2:
            weak_visual_embeddings = tf.expand_dims(weak_visual_embeddings, axis=1)
        if len(strong_textual_embeddings.shape) == 2:
            strong_textual_embeddings = tf.expand_dims(
                strong_textual_embeddings, axis=1
            )
        if len(weak_textual_embeddings.shape) == 2:
            weak_textual_embeddings = tf.expand_dims(weak_textual_embeddings, axis=1)

        fused_visual_contextualized_by_text = self.visual_to_text_fusion_block(
            query_features=strong_visual_embeddings,
            key_value_features=weak_textual_embeddings,
            training=training,
        )

        fused_textual_contextualized_by_visual = self.text_to_visual_fusion_block(
            query_features=strong_textual_embeddings,
            key_value_features=weak_visual_embeddings,
            training=training,
        )

        final_multimodal_embeddings = self.concatenation_layer(
            [
                fused_visual_contextualized_by_text,
                fused_textual_contextualized_by_visual,
            ]
        )

        final_multimodal_embeddings = tf.squeeze(final_multimodal_embeddings, axis=1)

        return final_multimodal_embeddings
