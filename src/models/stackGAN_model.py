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
from tensorflow.keras.layers import Input, Dense, Reshape, Concatenate
from tensorflow.keras.models import Model

latent_dim = 100  # Dimensionality of the noise vector for GAN
text_embedding_dim = 100  # Dimensionality of text embeddings (adjust based on your Word2Vec model)

# Stage-I Generator
def build_stage1_generator(latent_dim, text_embedding_dim):
    noise_input = Input(shape=(latent_dim,))
    text_input = Input(shape=(text_embedding_dim,))

    # Concatenate noise and text embeddings
    combined_input = Concatenate()([noise_input, text_input])

    x = Dense(256, activation='relu')(combined_input)
    x = Dense(512, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(64 * 64 * 3, activation='tanh')(x)  # Output shape for Stage-I generator

    generator = Model(inputs=[noise_input, text_input], outputs=output, name='stage1_generator')
    return generator

# Instantiate Stage-I generator
stage1_generator = build_stage1_generator(latent_dim, text_embedding_dim)

# Print model summary (optional)
stage1_generator.summary()

# Stage-II Generator
def build_stage2_generator(text_embedding_dim):
    text_input = Input(shape=(text_embedding_dim,))
    conditioning_input = Input(shape=(64, 64, 3))  # Output shape from Stage-I generator

    # Define model architecture for Stage-II generator
    # Adjust architecture based on StackGAN specifications for high-resolution image generation
    ...

    generator = Model(inputs=[text_input, conditioning_input], outputs=output, name='stage2_generator')
    return generator

# Instantiate Stage-II generator
stage2_generator = build_stage2_generator(text_embedding_dim)

# Print model summary (optional)
stage2_generator.summary()
