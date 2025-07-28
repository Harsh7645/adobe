from typing import List, Dict
from collections import Counter
import math

def score_section(section_text: str, keywords: List[str]) -> float:
    # Simple keyword count scoring
    text = section_text.lower()
    score = sum(text.count(kw) for kw in keywords)
    # Optionally, factor in length (log scale)
    score = score * math.log(len(section_text) + 1)
    return score

def rank_sections(sections: List[Dict], keywords: List[str], top_n: int = 5) -> List[Dict]:
    # Each section: {'title': ..., 'text': ...}
    for section in sections:
        section['score'] = score_section(section['text'], keywords)
    ranked = sorted(sections, key=lambda s: s['score'], reverse=True)
    return ranked[:top_n]
