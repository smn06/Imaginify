import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, BatchNormalization, Concatenate, Embedding, Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10  # Example dataset
import os
import datetime
import matplotlib.pyplot as plt

# Define hyperparameters
z_dim = 100         # Dimension of the noise vector (latent space)
num_classes = 10    # Number of classes for conditional input
emb_dim = 50        # Dimension of the embedding for conditional input
image_size = (64, 64, 3)  # Size of the generated image (e.g., 64x64 RGB)

# Path to save generated images and models
output_dir = './output_stackgan/'
os.makedirs(output_dir, exist_ok=True)

# Define input layers
z_input = Input(shape=(z_dim,), name='z_input')
c_input = Input(shape=(1,), dtype='int32', name='c_input')
image_input = Input(shape=image_size, name='image_input')

# Generator Architecture
def build_generator(z_input, c_input, image_size):
    # Conditional Augmentation Stage
    c_input_emb = Embedding(input_dim=num_classes, output_dim=emb_dim)(c_input)
    c_input_emb = Dense(units=z_dim)(c_input_emb)
    z = Concatenate()([z_input, c_input_emb])
    
    # Stage-I Generator
    g1 = Dense(1024, activation='relu')(z)
    g1 = BatchNormalization()(g1)
    g1 = Dense(128 * 16 * 16, activation='relu')(g1)
    g1 = BatchNormalization()(g1)
    g1 = Reshape((16, 16, 128))(g1)
    
    g1 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(g1)
    g1 = BatchNormalization()(g1)
    g1 = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh')(g1)  # Final output image

    return Model(inputs=[z_input, c_input], outputs=g1, name='Stage-I Generator')

# Discriminator Architecture
def build_discriminator(image_input, c_input, image_size):
    # ...

    return Model(inputs=[image_input, c_input], outputs=d_output, name='Discriminator')

# Build and compile models
generator_stage1 = build_generator(z_input, c_input, image_size)
generator_stage2 = build_generator(z_input, c_input, image_size)
discriminator = build_discriminator(image_input, c_input, image_size)

# Define optimizers
generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

# Compile models
generator_stage1.compile(optimizer=generator_optimizer, loss='binary_crossentropy')
generator_stage2.compile(optimizer=generator_optimizer, loss='binary_crossentropy')
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# Load and preprocess dataset (example with CIFAR-10)
(train_images, train_labels), (_, _) = cifar10.load_data()
train_images = (train_images - 127.5) / 127.5  # Normalize images to [-1, 1]
train_labels = train_labels.reshape(-1, 1)

# Training parameters
batch_size = 64
num_epochs = 100
save_interval = 10  # Save generated images and models every `save_interval` epochs
steps_per_epoch = len(train_images) // batch_size

# Training loop
for epoch in range(num_epochs):
    for step in range(steps_per_epoch):
        # Select a random batch of images and labels
        idx = np.random.randint(0, train_images.shape[0], batch_size)
        real_images = train_images[idx]
        real_labels = train_labels[idx]

        # Generate random noise and random labels
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        labels = np.random.randint(0, num_classes, (batch_size, 1))

        # Generate fake images using Stage-I Generator
        fake_images_stage1 = generator_stage1.predict([noise, labels])

        # Train Discriminator
        d_loss_real = discriminator.train_on_batch([real_images, real_labels], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([fake_images_stage1, labels], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator (Stage-I)
        g_loss_stage1 = generator_stage1.train_on_batch([noise, labels], np.ones((batch_size, 1)))

        # Train Generator (Stage-II)
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        fake_images_stage2 = generator_stage2.predict([noise, labels])

        # Train Discriminator (again with Stage-II generated images)
        d_loss_real = discriminator.train_on_batch([real_images, real_labels], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([fake_images_stage2, labels], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator (Stage-II)
        g_loss_stage2 = generator_stage2.train_on_batch([noise, labels], np.ones((batch_size, 1)))

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{steps_per_epoch}, "
              f"Generator Loss (Stage-I): {g_loss_stage1}, Generator Loss (Stage-II): {g_loss_stage2}, "
              f"Discriminator Loss: {d_loss}")

    # Save generated images and models periodically
    if epoch % save_interval == 0:
        # Generate images using test noise and labels
        test_noise = np.random.normal(0, 1, (num_classes, z_dim))
        test_labels = np.arange(num_classes).reshape(-1, 1)
        generated_images = generator_stage2.predict([test_noise, test_labels])

        # Plot and save generated images
        fig, axs = plt.subplots(num_classes, 1, figsize=(10, 2*num_classes))
        for i in range(num_classes):
            axs[i].imshow((generated_images[i] + 1) / 2)
            axs[i].axis('off')
            axs[i].set_title(f'Class {i}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'epoch_{epoch}_generated_images.png'))
        plt.close(fig)

        # Save models
        generator_stage1.save(os.path.join(output_dir, f'generator_stage1_epoch_{epoch}.h5'))
        generator_stage2.save(os.path.join(output_dir, f'generator_stage2_epoch_{epoch}.h5'))
        discriminator.save(os.path.join(output_dir, f'discriminator_epoch_{epoch}.h5'))

        # Print message
        print(f"Saved generated images and models at epoch {epoch}")

# Final save of models
generator_stage1.save(os.path.join(output_dir, 'final_generator_stage1.h5'))
generator_stage2.save(os.path.join(output_dir, 'final_generator_stage2.h5'))
discriminator.save(os.path.join(output_dir, 'final_discriminator.h5'))

print("Training completed.")
