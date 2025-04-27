import os
import pickle
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from newsapi import NewsApiClient
from typing import List, Dict, Tuple, Optional
import numpy as np
from dotenv import load_dotenv

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class NewsIREngine:
    def __init__(self):
        """Initialize the IR engine with necessary components."""
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.articles = []
        self.article_vectors = None
        
        # Initialize NewsAPI if API key is available
        load_dotenv()
        self.api_key = os.getenv('NEWS_API_KEY')
        if self.api_key:
            self.newsapi = NewsApiClient(api_key=self.api_key)
            print("Using NewsAPI with provided key")
        else:
            print("No NewsAPI key found. Using mock data.")

    def expand_query(self, query: str) -> str:
        """
        Expand the query using WordNet synonyms.
        Args:
            query (str): Original user query
        Returns:
            str: Expanded query with synonyms
        """
        tokens = word_tokenize(query.lower())
        expanded_terms = set()

        for token in tokens:
            expanded_terms.add(token)
            # Get synsets for the token
            synsets = wordnet.synsets(token)
            for syn in synsets[:2]:  # Limit to top 2 synsets to avoid noise
                for lemma in syn.lemmas():
                    expanded_terms.add(lemma.name().lower())

        return ' '.join(list(expanded_terms))

    def fetch_news(self, query: str) -> List[Dict]:
        """
        Fetch news articles using NewsAPI or return mock data.
        Args:
            query (str): Search query
        Returns:
            List[Dict]: List of news articles
        """
        if self.api_key:
            try:
                response = self.newsapi.get_everything(
                    q=query,
                    language='en',
                    sort_by='relevancy',
                    page_size=20
                )
                return response['articles']
            except Exception as e:
                print(f"Error fetching news: {e}")
                return self._get_mock_articles()
        else:
            return self._get_mock_articles()

    def _get_mock_articles(self) -> List[Dict]:
        """Return mock articles for testing."""
        return [
            {
                'title': 'AI Makes Breakthrough in Quantum Computing',
                'description': 'Scientists achieve major milestone in quantum computing using AI.',
                'url': 'https://example.com/article1'
            },
            {
                'title': 'New Study Shows Benefits of Remote Work',
                'description': 'Research indicates increased productivity in remote work settings.',
                'url': 'https://example.com/article2'
            },
            {
                'title': 'Climate Change Impact on Global Economy',
                'description': 'Economic experts analyze the effects of climate change on markets.',
                'url': 'https://example.com/article3'
            }
        ]

    def process_articles(self, articles: List[Dict]) -> None:
        """
        Process and vectorize articles using TF-IDF.
        Args:
            articles (List[Dict]): List of news articles
        """
        self.articles = articles
        texts = [f"{article['title']} {article['description']}" for article in articles]
        self.article_vectors = self.vectorizer.fit_transform(texts)

    def rank_articles(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Rank articles based on cosine similarity with the query.
        Args:
            query (str): User query
            top_k (int): Number of top results to return
        Returns:
            List[Tuple[Dict, float]]: Top-k articles with their similarity scores
        """
        if not self.articles or self.article_vectors is None:
            return []

        # Vectorize the expanded query
        expanded_query = self.expand_query(query)
        query_vector = self.vectorizer.transform([expanded_query])

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.article_vectors).flatten()
        
        # Get top-k articles
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [(self.articles[i], float(similarities[i])) for i in top_indices]

    def save_model(self, filepath: str = 'ir_model.pkl') -> None:
        """Save the vectorizer and other necessary objects."""
        model_data = {
            'vectorizer': self.vectorizer,
            'lemmatizer': self.lemmatizer
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, filepath: str = 'ir_model.pkl') -> 'NewsIREngine':
        """Load a saved model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls()
        instance.vectorizer = model_data['vectorizer']
        instance.lemmatizer = model_data['lemmatizer']
        return instance 