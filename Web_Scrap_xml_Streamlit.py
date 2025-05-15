import streamlit as st
import os
import re
import string
import nltk
import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from io import StringIO

# Ensure necessary downloads
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load resources only once
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()
lemmatizer = WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words("english")) | set(string.punctuation)

# Preprocess each XML content
def extract_text_from_xml(xml_content):
    try:
        soup = BeautifulSoup(xml_content, "lxml-xml")
        paragraphs = soup.find_all("para")
        return " ".join(p.text for p in paragraphs)
    except Exception as e:
        st.error(f"Error parsing XML: {e}")
        return ""

# Clean and normalize text
def clean_text(text):
    doc = nlp(text)
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != "-PRON-"]
    tokens = [tok for tok in tokens if tok not in stopwords and tok.isalpha() and len(tok) > 3]
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return " ".join(tokens)

# App UI
st.title("🧠 NLP XML Document Clustering")
st.markdown("Upload multiple XML files and cluster them using KMeans.")

uploaded_files = st.file_uploader("Upload XML files", accept_multiple_files=True, type=["xml"])
num_clusters = st.sidebar.slider("Select number of clusters", 2, 10, 6)
show_wordclouds = st.sidebar.checkbox("Show Wordclouds for Each Cluster", value=True)

if uploaded_files:
    raw_texts = [extract_text_from_xml(file.read().decode("utf-8")) for file in uploaded_files]
    clean_texts = [clean_text(text) for text in raw_texts]

    # Vectorization
    vectorizer = CountVectorizer(stop_words='english')
    count_matrix = vectorizer.fit_transform(clean_texts)
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)
    tfidf_matrix = normalize(tfidf_matrix)

    # Clustering
    model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=10, random_state=42)
    labels = model.fit_predict(tfidf_matrix)

    # Show result
    df = pd.DataFrame({'text': clean_texts, 'cluster': labels})
    st.dataframe(df)

    if show_wordclouds:
        for cluster in range(num_clusters):
            st.markdown(f"### Cluster {cluster}")
            cluster_words = " ".join(df[df['cluster'] == cluster]['text'])
            if cluster_words.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_words)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.info("No content to display in this cluster.")

    # Export CSV
    st.download_button("Download Clustered Data as CSV", df.to_csv(index=False), "clustered_data.csv", "text/csv")
else:
    st.info("Please upload at least one XML file.")
