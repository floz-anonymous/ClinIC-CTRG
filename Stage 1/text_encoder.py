import tensorflow as tf
from transformers import TFBertModel, TFAutoModel


class DualPathTextEncoder(tf.keras.Model):
    def __init__(self, embedding_dim=768):
        super(DualPathTextEncoder, self).__init__()

        self.strong_path_encoder = TFBertModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT", from_pt=True
        )
        self.strong_projector = tf.keras.layers.Dense(embedding_dim)

        self.weak_path_encoder = TFAutoModel.from_pretrained(
            "google/siglip-base-patch16-224", from_pt=True
        )
        self.weak_projector = tf.keras.layers.Dense(embedding_dim)

    def call(self, input_ids_strong, attention_mask_strong, input_ids_weak):
        strong_outputs = self.strong_path_encoder(
            input_ids=input_ids_strong, attention_mask=attention_mask_strong
        )
        f_s_t_features = strong_outputs.last_hidden_state[:, 0, :]
        f_s_t = self.strong_projector(f_s_t_features)

        weak_outputs = self.weak_path_encoder.text_model(input_ids=input_ids_weak)
        f_w_t_features = weak_outputs.pooler_output
        f_w_t = self.weak_projector(f_w_t_features)

        return f_s_t, f_w_t
