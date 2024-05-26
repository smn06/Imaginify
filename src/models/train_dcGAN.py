import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Assuming you have defined your DCGAN class in dcgan.py
from dcgan import DCGAN

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define parameters
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100
epochs = 20000
batch_size = 128
save_interval = 1000

# Load MNIST dataset
(X_train, _), (_, _) = mnist.load_data()

# Rescale -1 to 1
X_train = X_train / 127.5 - 1.0
X_train = np.expand_dims(X_train, axis=-1)

# Build and compile the DCGAN
dcgan = DCGAN(img_shape, latent_dim)

# Training function
def train_dcgan(dcgan, epochs, batch_size, save_interval):
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Sample noise and generate a batch of new images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = dcgan.generator.predict(noise)

        # Train the discriminator (real classified as ones and generated as zeros)
        d_loss_real = dcgan.discriminator.train_on_batch(imgs, valid)
        d_loss_fake = dcgan.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator (wants discriminator to mistake images as real)
        g_loss = dcgan.combined.train_on_batch(noise, valid)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, [D loss: {d_loss[0]}] [G loss: {g_loss}]")

        # If at save interval, save generated image samples
        if epoch % save_interval == 0:
            save_generated_images(epoch, dcgan)

def save_generated_images(epoch, dcgan, examples=10):
    r, c = 2, 5  # Grid size for displaying generated images
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = dcgan.generator.predict(noise)  # Generate images

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(f"generated_images/{epoch}.png")
    plt.close()

# Create directory for saving generated images
import os
os.makedirs("generated_images", exist_ok=True)

# Start training
train_dcgan(dcgan, epochs=epochs, batch_size=batch_size, save_interval=save_interval)
