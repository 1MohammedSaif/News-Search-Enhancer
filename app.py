import streamlit as st
import os
from ir_engine import NewsIREngine
from typing import Optional

def initialize_session_state():
    """Initialize session state variables."""
    if 'ir_engine' not in st.session_state:
        st.session_state.ir_engine = NewsIREngine()
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

def display_article(article: dict, similarity_score: float):
    """Display a single article with its details."""
    st.markdown(f"### [{article['title']}]({article['url']})")
    st.markdown(f"**Relevance Score:** {similarity_score:.2f}")
    st.markdown(f"**Description:** {article['description']}")
    st.markdown("---")

def main():
    st.set_page_config(
        page_title="News Search Enhancer",
        page_icon="📰",
        layout="wide"
    )

    st.title("📰 News Search Enhancer")
    st.markdown("""
    This application uses Information Retrieval techniques to find and rank relevant news articles.
    Enter your search query below to get started!
    """)

    # Initialize session state
    initialize_session_state()

    # Sidebar controls
    with st.sidebar:
        st.header("Search Settings")
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        use_query_expansion = st.checkbox("Enable Query Expansion", value=True)
        
        st.header("Search History")
        for query in st.session_state.search_history:
            st.text(f"• {query}")

    # Main search interface
    query = st.text_input("Enter your search query:", key="search_query")
    
    if st.button("Search") and query:
        try:
            with st.spinner("Searching for articles..."):
                # Add query to search history
                if query not in st.session_state.search_history:
                    st.session_state.search_history.append(query)
                
                # Fetch and process articles
                articles = st.session_state.ir_engine.fetch_news(query)
                if not articles:
                    st.warning("No articles found. Please try a different search query.")
                    return

                st.session_state.ir_engine.process_articles(articles)
                
                # Get ranked results
                ranked_articles = st.session_state.ir_engine.rank_articles(
                    query if not use_query_expansion else st.session_state.ir_engine.expand_query(query),
                    top_k=top_k
                )

                # Display results
                if ranked_articles:
                    st.subheader(f"Found {len(ranked_articles)} relevant articles:")
                    for article, score in ranked_articles:
                        display_article(article, score)
                else:
                    st.warning("No relevant articles found for your query.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try again with a different query or check your internet connection.")

    # Display API status
    if 'NEWS_API_KEY' not in os.environ:
        st.warning("""
        ⚠️ NewsAPI key not found. Using mock data for demonstration.
        To use real news data, please set your NEWS_API_KEY in a .env file.
        """)

if __name__ == "__main__":
    main() 