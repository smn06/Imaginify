import tensorflow as tf
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import os

# Paths to saved models and embeddings
stage1_generator_path = "./models/stackgan/stage1_generator.h5"
stage2_generator_path = "./models/stackgan/stage2_generator.h5"
embeddings_path = "./models/embeddings/word2vec.model"

# Load Word2Vec model for text embeddings
word2vec_model = Word2Vec.load(embeddings_path)

# Load Stage-I and Stage-II generators
stage1_generator = load_model(stage1_generator_path)
stage2_generator = load_model(stage2_generator_path)

def generate_image_from_text(text, stage1_generator, stage2_generator, word2vec_model, latent_dim=100):
    # Tokenize and embed text
    tokens = word_tokenize(text.lower())
    embeddings = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    embedded_text = np.mean(embeddings, axis=0, keepdims=True)

    # Generate noise vector
    noise_vector = np.random.normal(0, 1, size=(1, latent_dim))

    # Generate Stage-I image
    stage1_input = [noise_vector, embedded_text]
    stage1_image = stage1_generator.predict(stage1_input)

    # Generate Stage-II image
    stage2_input = [embedded_text, stage1_image]
    generated_image = stage2_generator.predict(stage2_input)

    # Post-process generated image (optional)
    generated_image = (generated_image + 1) / 2.0  # Scale to [0, 1] if using tanh activation

    return generated_image

def save_image(image, save_path='./generated_images/', filename='generated_image.png'):
    os.makedirs(save_path, exist_ok=True)
    image_path = os.path.join(save_path, filename)
    tf.keras.preprocessing.image.save_img(image_path, image[0])
    print(f"Generated image saved at: {image_path}")

# Example usage
text_description = "a yellow bird with red wings flying in sky"
generated_image = generate_image_from_text(text_description, stage1_generator, stage2_generator, word2vec_model)
save_image(generated_image)
