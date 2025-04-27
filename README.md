# News Search Enhancer

A Python-based news search application that uses Information Retrieval concepts to enhance search results. The application features query expansion using WordNet, TF-IDF vectorization, and cosine similarity for ranking articles.

## Features

- Query expansion using WordNet synonyms
- News article fetching using NewsAPI (or mock data)
- TF-IDF vectorization for text representation
- Cosine similarity-based article ranking
- Streamlit-based user interface
- Search history tracking
- Configurable number of results
- Toggle for query expansion

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd news-search-enhancer
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. (Optional) Set up NewsAPI:
   - Get an API key from [NewsAPI](https://newsapi.org/)
   - Create a `.env` file in the project root
   - Add your API key: `NEWS_API_KEY=your_api_key_here`

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)

3. Enter your search query in the text input field

4. Adjust search settings in the sidebar:
   - Number of results to display
   - Toggle query expansion

5. Click "Search" to see the results

## Project Structure

- `ir_engine.py`: Core IR functionality (query expansion, vectorization, ranking)
- `app.py`: Streamlit frontend application
- `requirements.txt`: Project dependencies
- `.env`: (Optional) Environment variables for API keys

## Notes

- If no NewsAPI key is provided, the application will use mock data for demonstration
- Query expansion can be toggled on/off in the sidebar
- Search history is maintained during the session
- Results are ranked by relevance using cosine similarity

## Dependencies

- streamlit
- nltk
- scikit-learn
- requests
- numpy
- pandas
- python-dotenv
- newsapi-python

## License

This project is licensed under the MIT License - see the LICENSE file for details. 