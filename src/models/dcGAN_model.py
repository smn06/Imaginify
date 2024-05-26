from gensim.models import Word2Vec
import numpy as np

# Load Word2Vec model
embeddings_path = "./models/embeddings/word2vec.model"
word2vec_model = Word2Vec.load(embeddings_path)

# Function to embed text input
def embed_text(text):
    tokens = word_tokenize(text.lower())
    embeddings = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    # Aggregate embeddings (e.g., averaging)
    embedded_text = np.mean(embeddings, axis=0)
    return embedded_text

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model

latent_dim = 100  # Dimensionality of the noise vector for GAN
img_shape = (28, 28, 1)  # Image shape (adjust according to dataset)

# Generator Model
def build_generator(latent_dim, text_embedding_dim):
    noise_input = Input(shape=(latent_dim,))
    text_input = Input(shape=(text_embedding_dim,))  # Adjust dimensionality based on embeddings

    # Merge noise and text embeddings
    combined_input = Concatenate()([noise_input, text_input])

    x = Dense(256 * 7 * 7, activation='relu')(combined_input)
    x = Reshape((7, 7, 256))(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    output = Conv2DTranspose(1, kernel_size=5, strides=1, padding='same', activation='sigmoid')(x)

    generator = Model(inputs=[noise_input, text_input], outputs=output, name='generator')
    return generator

# Discriminator Model
def build_discriminator(img_shape, text_embedding_dim):
    img_input = Input(shape=img_shape)

    x = Conv2D(64, kernel_size=5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(img_input)
    x = Conv2D(128, kernel_size=5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(x)
    x = Flatten()(x)

    text_input = Input(shape=(text_embedding_dim,))  # Adjust dimensionality based on embeddings
    combined_input = Concatenate()([x, text_input])

    x = Dense(1, activation='sigmoid')(combined_input)

    discriminator = Model(inputs=[img_input, text_input], outputs=x, name='discriminator')
    return discriminator

# Build and compile the GAN
def build_gan(generator, discriminator):
    noise_input = Input(shape=(latent_dim,))
    text_input = Input(shape=(text_embedding_dim,))  # Adjust dimensionality based on embeddings

    generated_img = generator([noise_input, text_input])
    gan_output = discriminator([generated_img, text_input])

    gan = Model(inputs=[noise_input, text_input], outputs=gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan

# Instantiate models
generator = build_generator(latent_dim, 100)  # Adjust text_embedding_dim based on your Word2Vec model
discriminator = build_discriminator(img_shape, 100)  # Adjust text_embedding_dim based on your Word2Vec model
gan = build_gan(generator, discriminator)

# Print model summary (optional)
generator.summary()
discriminator.summary()
gan.summary()

