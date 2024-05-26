import os
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Paths
DATA_PATH = "./data/raw/text_descriptions.csv"
PROCESSED_DATA_PATH = "./data/processed/processed_dataset.csv"
EMBEDDINGS_PATH = "./models/embeddings/word2vec.model"
EMBEDDING_SIZE = 100  # Size of the word embeddings

def load_data(data_path):
    """
    Load the textual data from a CSV file.
    """
    data = pd.read_csv(data_path)
    return data

def preprocess_text(text):
    """
    Preprocess the text by tokenizing and cleaning.
    """
    tokens = word_tokenize(text.lower())
    return tokens

def build_word2vec_model(sentences, embedding_size):
    """
    Train a Word2Vec model on the given sentences.
    """
    model = Word2Vec(sentences=sentences, vector_size=embedding_size, window=5, min_count=1, workers=4)
    return model

def save_word2vec_model(model, path):
    """
    Save the Word2Vec model to the specified path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

def generate_text_embeddings(data, model):
    """
    Generate text embeddings for each description in the dataset.
    """
    embeddings = []
    for description in data['clean_description']:
        tokens = preprocess_text(description)
        word_embeddings = [model.wv[word] for word in tokens if word in model.wv]
        if word_embeddings:
            sentence_embedding = np.mean(word_embeddings, axis=0)
        else:
            sentence_embedding = np.zeros(model.vector_size)
        embeddings.append(sentence_embedding)
    return np.array(embeddings)

def save_processed_data(data, embeddings, path):
    """
    Save the processed dataset with text embeddings.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    embedding_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(embeddings.shape[1])])
    processed_data = pd.concat([data, embedding_df], axis=1)
    processed_data.to_csv(path, index=False)

def main():
    # Step 1: Load raw data
    print("Loading raw data...")
    data = load_data(DATA_PATH)

    # Step 2: Preprocess text data
    print("Preprocessing text data...")
    data['clean_description'] = data['description'].apply(preprocess_text)

    # Step 3: Build Word2Vec model
    print("Building Word2Vec model...")
    sentences = data['clean_description'].tolist()
    model = build_word2vec_model(sentences, EMBEDDING_SIZE)

    # Step 4: Save Word2Vec model
    print("Saving Word2Vec model...")
    save_word2vec_model(model, EMBEDDINGS_PATH)

    # Step 5: Generate text embeddings
    print("Generating text embeddings...")
    embeddings = generate_text_embeddings(data, model)

    # Step 6: Save processed data
    print("Saving processed data with embeddings...")
    save_processed_data(data, embeddings, PROCESSED_DATA_PATH)

    print("Feature building complete!")

if __name__ == "__main__":
    main()
