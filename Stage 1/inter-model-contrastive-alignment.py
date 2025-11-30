import tensorflow as tf


class ProjectionHead(tf.keras.layers.Layer):
    def __init__(self, output_dim=256, hidden_dim=512):
        super(ProjectionHead, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(output_dim)
        self.norm = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.norm(x)


def info_nce_loss(query_embeddings, positive_embeddings, temperature=0.07):
    logits = (
        tf.matmul(query_embeddings, positive_embeddings, transpose_b=True) / temperature
    )
    batch_size = tf.shape(query_embeddings)[0]
    labels = tf.range(batch_size)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = loss_fn(labels, logits)
    return loss


class InterModalAlignment(tf.keras.Model):
    def __init__(self, projection_dim=256, temperature=0.07):
        super(InterModalAlignment, self).__init__()
        self.temperature = temperature

        self.proj_head_visual_strong = ProjectionHead(output_dim=projection_dim)
        self.proj_head_visual_weak = ProjectionHead(output_dim=projection_dim)

        self.proj_head_textual_strong = ProjectionHead(output_dim=projection_dim)
        self.proj_head_textual_weak = ProjectionHead(output_dim=projection_dim)

    def call(
        self,
        strong_visual_features,
        weak_visual_features,
        strong_text_features,
        weak_text_features,
    ):
        projected_strong_visual = self.proj_head_visual_strong(strong_visual_features)
        projected_weak_text = self.proj_head_textual_weak(weak_text_features)

        projected_strong_text = self.proj_head_textual_strong(strong_text_features)
        projected_weak_visual = self.proj_head_visual_weak(weak_visual_features)

        loss_visual_to_text = info_nce_loss(
            projected_strong_visual, projected_weak_text, self.temperature
        )

        loss_text_to_visual = info_nce_loss(
            projected_strong_text, projected_weak_visual, self.temperature
        )

        total_inter_modal_loss = loss_visual_to_text + loss_text_to_visual

        return total_inter_modal_loss
