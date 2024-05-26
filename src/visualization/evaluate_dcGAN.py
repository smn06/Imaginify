import tensorflow as tf
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import os

# Paths to saved models and embeddings
generator_path = "./models/dcgan/generator.h5"
embeddings_path = "./models/embeddings/word2vec.model"

# Load Word2Vec model for text embeddings
word2vec_model = Word2Vec.load(embeddings_path)

# Load DCGAN generator
generator = load_model(generator_path)

def generate_image_from_text(text, generator, word2vec_model, latent_dim=100):
    # Tokenize and embed text
    tokens = word_tokenize(text.lower())
    embeddings = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    embedded_text = np.mean(embeddings, axis=0, keepdims=True)

    # Generate noise vector
    noise_vector = np.random.normal(0, 1, size=(1, latent_dim))

    # Concatenate noise and embedded text
    input_vector = np.concatenate([noise_vector, embedded_text], axis=1)

    # Generate image
    generated_image = generator.predict(input_vector)

    # Post-process generated image (optional)
    generated_image = (generated_image + 1) / 2.0  # Scale to [0, 1] if using tanh activation

    return generated_image

def save_image(image, save_path='./generated_images/', filename='generated_image.png'):
    os.makedirs(save_path, exist_ok=True)
    image_path = os.path.join(save_path, filename)
    tf.keras.preprocessing.image.save_img(image_path, image[0])
    print(f"Generated image saved at: {image_path}")

# Example textual descriptions
textual_descriptions = [
    "a sunset over a calm lake",
    "a futuristic city skyline at night",
    "a forest with sunlight filtering through trees"
]

# Generate and save images for each description
for idx, text_description in enumerate(textual_descriptions):
    generated_image = generate_image_from_text(text_description, generator, word2vec_model)
    save_image(generated_image, save_path=f'./generated_images/', filename=f'generated_image_{idx}.png')


