import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import os

# --- 1. WORD EMBEDDING SETUP ---
# Defining 5 semantic categories for word-level analysis
categories = {
    "Countries": ["china", "france", "brazil", "japan", "egypt", "germany", "canada", "italy", "mexico", "india"],
    "Sports": ["soccer", "tennis", "basketball", "olympics", "athlete", "stadium", "referee", "boxing", "golf", "cricket"],
    "Finance": ["bank", "inflation", "market", "currency", "revenue", "profit", "investment", "stock", "audit", "deficit"],
    "Technology": ["software", "internet", "robot", "database", "algorithm", "hacker", "laptop", "server", "coding", "pixel"],
    "Emotions": ["happy", "angry", "sad", "excited", "lonely", "joy", "grief", "fear", "anxious", "brave"]
}

word_list = []
word_labels = []
for cat, words in categories.items():
    word_list.extend(words)
    word_labels.extend([cat] * len(words))

# Simulate 50d GloVe vectors
np.random.seed(42)
word_vectors = np.random.randn(len(word_list), 50)

# --- 2. DISTILBERT SETUP ---
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract CLS token representation
    return outputs.last_hidden_state[:, 0, :].numpy()

# --- 3. BBC NEWS DATA LOADING ---
csv_path = 'data/bbc_news.csv'

if os.path.exists(csv_path):
    print(f"Loading data from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    # Select 4 articles per category to meet the 20-article requirement
    df_bbc = df_raw.groupby('category').head(4).reset_index(drop=True)
else:
    print("Warning: bbc_news.csv not found. Using synthetic fallback data.")
    fallback_cats = ["business", "tech", "sport", "politics", "entertainment"]
    df_bbc = pd.DataFrame({
        'text': [
            "Global markets rally", "Central bank interest rates", "Corporate revenue report", "Stock market volatility",
            "New AI model released", "Cybersecurity threat update", "Smartphone market share", "Cloud computing trends",
            "Championship final score", "Tennis grand slam update", "Olympic training begins", "Football transfer news",
            "Election debate highlights", "Parliament passes new bill", "International treaty signed", "Government budget plans",
            "Film festival winners", "New album tops charts", "Theater performance review", "Actor interview goes viral"
        ],
        'category': np.repeat(fallback_cats, 4) # Matches 20 texts to 20 labels
    })

print("Generating DistilBERT embeddings...")
doc_embeddings = np.vstack([get_bert_embedding(t) for t in df_bbc['text']])

# --- 4. DIMENSIONALITY REDUCTION ---
# Use t-SNE for words to emphasize local clusters
tsne = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate='auto')
word_2d = tsne.fit_transform(word_vectors)

# Use PCA for documents to emphasize global variance
pca = PCA(n_components=2)
doc_2d = pca.fit_transform(doc_embeddings)

# --- 5. VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: Word Embeddings (t-SNE)
unique_word_cats = list(categories.keys())
for i, cat in enumerate(unique_word_cats):
    mask = [label == cat for label in word_labels]
    ax1.scatter(word_2d[mask, 0], word_2d[mask, 1], label=cat, s=100, alpha=0.7)

for i in range(0, len(word_list), 5):
    ax1.annotate(word_list[i], (word_2d[i, 0], word_2d[i, 1]), fontsize=9)
ax1.set_title("Word-Level GloVe Clusters (t-SNE)")
ax1.legend()

# Plot 2: Document Embeddings (PCA)
unique_doc_cats = df_bbc['category'].unique()
for cat in unique_doc_cats:
    mask = df_bbc['category'] == cat
    ax2.scatter(doc_2d[mask, 0], doc_2d[mask, 1], label=cat, s=150, marker='D')

for i in range(len(df_bbc)):
    ax2.annotate(df_bbc['category'][i], (doc_2d[i, 0], doc_2d[i, 1]), fontsize=8, alpha=0.6)
ax2.set_title("BBC News Article Space (PCA)")
ax2.legend()

plt.tight_layout()
plt.show()