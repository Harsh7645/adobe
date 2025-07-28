from typing import List, Dict, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string
from sentence_transformers import SentenceTransformer

class KeywordExtractor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['would', 'could', 'should', 'might', 'must', 'need', 'shall'])
        
        # Initialize sentence transformer model for semantic similarity
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract most relevant keywords from text"""
        # Tokenize and clean
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens 
                 if token not in self.stop_words 
                 and token not in string.punctuation
                 and len(token) > 2]
        
        # Count word frequencies
        word_freq = Counter(tokens)
        
        # Get top N keywords by frequency
        keywords = [word for word, _ in word_freq.most_common(top_n)]
        return keywords
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using sentence transformer"""
        try:
            # Encode both texts
            embeddings = self.model.encode([text1, text2])
            # Calculate cosine similarity using dot product
            similarity = (embeddings[0] * embeddings[1]).sum()
            return float(similarity)
        except Exception:
            return 0.0
    
    def add_domain_keywords(self, keywords: List[str], domain: str) -> List[str]:
        """Add domain-specific keywords based on the context"""
        domain_keywords = {
            'technical': ['implementation', 'architecture', 'system', 'design', 'requirements'],
            'business': ['strategy', 'market', 'revenue', 'growth', 'opportunity'],
            'tourism': ['attractions', 'accommodation', 'activities', 'travel', 'experience', 'food', 'local', 'culture'],
            'education': ['learning', 'curriculum', 'students', 'teaching', 'assessment']
        }
        
        if domain.lower() in domain_keywords:
            keywords.extend(domain_keywords[domain])
        
        return list(set(keywords))  # Remove duplicates
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Alias for calculate_semantic_similarity"""
        return self.calculate_semantic_similarity(text1, text2)
