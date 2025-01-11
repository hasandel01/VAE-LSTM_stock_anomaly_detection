import tensorflow as tf
import tensorflow.keras.layers as layers

class VAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim=2, hidden_dim=32):
        super(VAE,self).__init__()
        self.latent_dim = latent_dim

        #Encoder part.
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(latent_dim * 2) ## OUTPUT: Z_MEAN & Z_LOGVAR
        ])

        #Decoder part.
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')
        ])

    def encode(self, x):
        z_mean_logvar = self.encoder(x)
        z_mean = z_mean_logvar[:, :self.latent_dim]
        z_logvar = z_mean_logvar[:, self.latent_dim:]
        return z_mean, z_logvar

    def reparameterize(self, z_mean, z_logvar):
        eps = tf.random.normal(shape=z_mean.shape)
        return z_mean + tf.exp(0.5 * z_logvar) * eps

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        x_recon = self.decode(z)
        return x_recon, z_mean, z_logvar


def vae_loss(x, x_recon, z_mean, z_logvar):

    recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_recon), axis=1))

    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1)
    )
    return recon_loss + kl_loss


def train_vae(vae, train_data, batch_size=64, epochs=30, learning_rate=0.001):
    """
    Train variational autoencoder model and capture loss components over epochs.
    :param vae: Model to be trained.
    :param train_data: Training data.
    :param batch_size: Batch size.
    :param epochs: Number of epochs.
    :param learning_rate: Learning rate.
    :return: Lists of reconstruction loss, KL divergence loss, and total loss per epoch.
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    dataset = tf.data.Dataset.from_tensor_slices(
        tf.cast(train_data, tf.float32)
    ).batch(batch_size)

    # Lists to store loss components for each epoch
    reconstruction_losses = []
    kl_losses = []
    total_losses = []

    for epoch in range(epochs):
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_total_loss = 0
        for step, x_batch in enumerate(dataset):
            with tf.GradientTape() as tape:
                x_recon, z_mean, z_logvar = vae(x_batch)

                # Compute losses
                recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_batch - x_recon), axis=1))
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1)
                )
                total_loss = recon_loss + kl_loss

                # Accumulate losses for the epoch
                epoch_recon_loss += recon_loss.numpy()
                epoch_kl_loss += kl_loss.numpy()
                epoch_total_loss += total_loss.numpy()

            # Backpropagation
            grads = tape.gradient(total_loss, vae.trainable_variables)
            optimizer.apply_gradients(zip(grads, vae.trainable_variables))

        # Average losses for the epoch
        reconstruction_losses.append(epoch_recon_loss / (step + 1))
        kl_losses.append(epoch_kl_loss / (step + 1))
        total_losses.append(epoch_total_loss / (step + 1))

        print(f'Epoch {epoch + 1}/{epochs}, Total Loss: {total_losses[-1]}, '
                  f'Recon Loss: {reconstruction_losses[-1]}, KL Loss: {kl_losses[-1]}')

    return reconstruction_losses, kl_losses, total_losses
