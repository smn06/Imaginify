import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os

nltk.download('punkt')
nltk.download('stopwords')

data_path = "./data/raw/dataset.csv"  # Path to raw dataset file
save_path = "./data/processed/"       # Directory to save processed data

# Load dataset
df = pd.read_csv(data_path)

# Clean textual descriptions
def clean_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Apply cleaning function
df['clean_description'] = df['description'].apply(clean_text)

# Save processed data
os.makedirs(save_path, exist_ok=True)
df.to_csv(os.path.join(save_path, "processed_dataset.csv"), index=False)
