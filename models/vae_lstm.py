import tensorflow as tf

class VAELSTMHybrid(tf.keras.Model):
    def __init__(self, input_dim, latent_dim=2, lstm_units=64, hidden_dim=32):
        super(VAELSTMHybrid, self).__init__()
        self.latent_dim = latent_dim

        # VAE Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(latent_dim * 2)  # z_mean and z_logvar
        ])

        # VAE Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='sigmoid')  # Ensure output matches input_dim
        ])

        # LSTM for Sequential Prediction
        self.lstm = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(None, latent_dim)),  # (time_steps, latent_dim)
            tf.keras.layers.LSTM(lstm_units, return_sequences=False),
            tf.keras.layers.Dense(1)  # Predict a single value
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

    def call(self, x, lstm_input=None):
        # VAE Encoding
        batch_size, time_steps, features = x.shape
        x_flat = tf.reshape(x, (-1, features))  # Flatten sequence dimension
        z_mean, z_logvar = self.encode(x_flat)
        z = self.reparameterize(z_mean, z_logvar)

        # LSTM Prediction
        lstm_input = z  # Use the latent embeddings as LSTM input
        if lstm_input is not None:
            lstm_input = tf.reshape(lstm_input, (-1, time_steps, self.latent_dim))  # Adjust `time_steps` accordingly
            lstm_output = self.lstm(lstm_input)
        else:
            lstm_output = None

        # Decoding from predicted embedding
        x_recon = self.decode(z)
        x_recon = tf.reshape(x_recon, (batch_size, time_steps, features))  # Reshape to match X_test

        return x_recon, z_mean, z_logvar, lstm_output


def vae_lstm_loss(x, x_recon, z_mean, z_logvar, y_true, y_pred, beta=1.0):
    # Ensure consistent data types for `x` and `x_recon`
    x = tf.cast(x, tf.float32)
    x_recon = tf.cast(x_recon, tf.float32)

    # VAE Reconstruction Loss
    recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_recon), axis=1))

    # VAE KL Divergence Loss
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1)
    )

    # LSTM Loss
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    lstm_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    return recon_loss + beta * kl_loss + lstm_loss



def train_vae_lstm_hybrid(model, train_data, lstm_train_data, batch_size=64, epochs=30, learning_rate=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    dataset = tf.data.Dataset.from_tensor_slices((train_data, lstm_train_data)).batch(batch_size)

    for epoch in range(epochs):
        total_loss = 0
        for step, (batch_x, batch_lstm_input) in enumerate(dataset):
            with tf.GradientTape() as tape:
                x_recon, z_mean, z_logvar, lstm_output = model(batch_x, lstm_input=batch_lstm_input)
                loss = vae_lstm_loss(batch_x, x_recon, z_mean, z_logvar, batch_lstm_input[:, -1, 0], lstm_output)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += loss.numpy()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (step + 1)}")
