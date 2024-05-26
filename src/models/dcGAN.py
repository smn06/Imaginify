import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam

class DCGAN:
    def __init__(self, img_shape, latent_dim):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.optimizer = Adam(lr=0.0002, beta_1=0.5)

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        z = tf.keras.Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        validity = self.discriminator(img)

        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation="relu"))
        model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding="same", activation="relu"))
        model.add(Conv2DTranspose(self.img_shape[2], kernel_size=3, strides=2, padding='same', activation='tanh'))

        model.summary()

        noise = tf.keras.Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = tf.keras.Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)


# Assuming img_shape and latent_dim are defined appropriately
img_shape = (64, 64, 3)  # Example image shape
latent_dim = 100  # Example latent dimension

dcgan = DCGAN(img_shape, latent_dim)


def train_dcgan(dcgan, dataset, epochs, batch_size=32, save_interval=50):
    # Implement your training loop here
    for epoch in range(epochs):
        # Sample batch of images from dataset
        batch_images = dataset[np.random.randint(0, dataset.shape[0], batch_size)]

        # Generate noise input for generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Generate images using generator
        gen_images = dcgan.generator.predict(noise)

        # Train discriminator on real and generated images
        d_loss_real = dcgan.discriminator.train_on_batch(batch_images, np.ones((batch_size, 1)))
        d_loss_fake = dcgan.discriminator.train_on_batch(gen_images, np.zeros((batch_size, 1)))

        # Train generator (via combined model)
        g_loss = dcgan.combined.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print progress
        print(f"Epoch {epoch}, [D loss: {0.5 * np.add(d_loss_real, d_loss_fake)}] [G loss: {g_loss}]")

        # Save generated images at save_interval epochs
        if epoch % save_interval == 0:
            save_generated_images(epoch, dcgan)

def save_generated_images(epoch, dcgan, examples=10):
    # Generate images and save them
    noise = np.random.normal(0, 1, (examples, latent_dim))
    gen_images = dcgan.generator.predict(noise)

    # Save images
    for i in range(examples):
        plt.imshow(gen_images[i, :, :, :])
        plt.axis('off')
        plt.savefig(f"generated_image_{epoch}_{i}.png")
        plt.close()

# Example usage:
# train_dcgan(dcgan, dataset, epochs=10000, batch_size=32, save_interval=100)

