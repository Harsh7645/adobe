import re
from collections import Counter
from typing import List

# Optionally, you can use nltk or spacy for more advanced stopword removal
DEFAULT_STOPWORDS = set([
    'this', 'that', 'with', 'from', 'which', 'will', 'have', 'been', 'were', 'their', 'about', 'there',
    'your', 'more', 'than', 'when', 'where', 'what', 'such', 'shall', 'each', 'also', 'into', 'only',
    'other', 'some', 'most', 'must', 'very', 'should', 'could', 'would', 'upon', 'they', 'them', 'then',
    'these', 'those', 'over', 'under', 'after', 'before', 'because', 'while', 'between', 'among', 'being',
    'through', 'during', 'without', 'within', 'against', 'above', 'below', 'again', 'further', 'once',
    'same', 'just', 'like', 'even', 'many', 'much', 'every', 'both', 'any', 'all', 'our', 'out', 'off',
    'for', 'and', 'are', 'but', 'not', 'was', 'you', 'the', 'has', 'can', 'may', 'his', 'her', 'him',
    'she', 'who', 'its', 'had', 'did', 'how', 'why', 'use', 'used', 'using', 'get', 'got', 'let', 'lets',
    'see', 'see', 'here', 'page', 'pages', 'section', 'table', 'figure', 'figures', 'appendix', 'chapter',
    'introduction', 'summary', 'conclusion', 'background', 'abstract', 'overview', 'results', 'discussion',
    'references', 'acknowledgements', 'contents', 'history', 'version', 'document', 'title', 'syllabus', 'board'
])

def extract_keywords(text: str, top_n: int = 10, extra_keywords: List[str] = None) -> List[str]:
    words = re.findall(r'\b\w{4,}\b', text.lower())
    stopwords = DEFAULT_STOPWORDS.copy()
    if extra_keywords:
        stopwords -= set(extra_keywords)
    filtered = [w for w in words if w not in stopwords]
    freq = Counter(filtered)
    keywords = [w for w, _ in freq.most_common(top_n)]
    if extra_keywords:
        keywords = list(set(keywords + extra_keywords))
    return keywords
