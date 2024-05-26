import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import os

# Load processed dataset
data_path = "./data/processed/processed_dataset.csv"
df = pd.read_csv(data_path)

# Tokenize text
df['tokens'] = df['clean_description'].apply(word_tokenize)


# Train Word2Vec model
word2vec_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Save Word2Vec model
embeddings_path = "./models/embeddings/"
os.makedirs(embeddings_path, exist_ok=True)
word2vec_model.save(os.path.join(embeddings_path, "word2vec.model"))

