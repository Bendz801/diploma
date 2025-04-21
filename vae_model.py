import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer

class KLDivergenceLoss(Layer):
    def __init__(self, z_mean, z_log_var, **kwargs):
        super(KLDivergenceLoss, self).__init__(**kwargs)
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.kl_loss = self.add_weight(name='kl_loss', initializer='zeros', trainable=False)

    def call(self, inputs):
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(inputs, inputs))
        kl = -0.5 * tf.reduce_mean(1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var))
        self.kl_loss.assign(kl)
        self.add_loss(reconstruction_loss + kl)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

def build_vae(config):
    input_dim = config['vae']['input_dim']
    latent_dim = config['vae']['latent_dim']
    inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(64, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)

    def sampling(args):
        mean, log_var = args
        eps = tf.random.normal(shape=(tf.shape(mean)[0], latent_dim))
        return mean + tf.exp(0.5 * log_var) * eps

    z = layers.Lambda(sampling)([z_mean, z_log_var])
    kl_loss_layer = KLDivergenceLoss(z_mean=z_mean, z_log_var=z_log_var)(inputs)

    dec_h = layers.Dense(64, activation='relu')(z)
    outputs = layers.Dense(input_dim, activation='sigmoid')(dec_h)

    vae = Model(inputs=inputs, outputs=outputs)
    vae.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return vae

def train_vae(vae, data, config):
    # Assuming 'Close' is not relevant for VAE training based on your previous 'main.py'
    X = data
    if X.size > 0:
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8) # Adding a small epsilon
        vae.fit(X, X, # Autoencoder: target is the input itself
                epochs=config['vae']['epochs'],
                batch_size=config['vae']['batch_size'],
                verbose=1)
    else:
        print("Предупреждение: Нет данных для обучения VAE.")