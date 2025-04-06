# vae_model.py
import json
import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
import numpy as np

def build_vae(config):
    """
    Создаёт и обучает VAE.
    Args:
        config (dict): Конфигурация.
    Returns:
        Model: Обученная модель VAE.
    """
    input_dim = config["vae"]["input_dim"]
    latent_dim = config["vae"]["latent_dim"]
    epochs = config["vae"]["epochs"]
    batch_size = config["vae"]["batch_size"]

    # Энкодер
    inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(64, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)

    # Репараметризация
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_var])

    # Декодер
    decoder_h = layers.Dense(64, activation='relu')(z)
    outputs = layers.Dense(input_dim, activation='sigmoid')(decoder_h)

    # Модель
    vae = Model(inputs, outputs)

    # Потери
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(inputs, outputs))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)

    # Компиляция
    vae.compile(optimizer='adam')
    return vae

def train_vae(vae, data, config):
    """
    Обучает VAE.
    Args:
        vae (Model): Модель VAE.
        data (pd.DataFrame): Данные.
        config (dict): Конфигурация.
    """
    # Нормализация данных
    features = data.drop(columns=['Close']).values
    features = (features - features.min()) / (features.max() - features.min())

    # Обучение
    vae.fit(features, features, epochs=config["vae"]["epochs"], batch_size=config["vae"]["batch_size"], verbose=1)

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    data = pd.read_csv(config["data"]["processed_data_path"], parse_dates=['Date'], index_col='Date')
    vae = build_vae(config)
    train_vae(vae, data, config)