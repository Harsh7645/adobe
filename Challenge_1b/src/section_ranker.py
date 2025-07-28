from typing import List, Dict
import re
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class SectionRanker:
    def __init__(self, keyword_extractor):
        self.keyword_extractor = keyword_extractor
        self.tfidf = TfidfVectorizer(stop_words='english')
        
    def _determine_category(self, section: Dict, categories: Dict[str, List[str]]) -> str:
        """Determine the category of a section based on its content"""
        text = (section.get("title", "") + " " + section.get("text", "")).lower()
        scores = {}
        for category, keywords in categories.items():
            scores[category] = sum(keyword in text for keyword in keywords)
            
        max_category = max(scores.items(), key=lambda x: x[1])[0]
        return max_category if scores[max_category] > 0 else 'other'
        
    def rank_sections(self, sections: List[Dict], keywords: List[str]) -> List[Dict]:
        """Rank sections based on keyword relevance and content importance"""
        ranked_sections = []
        relevant_categories = {
            'cities': ['city', 'cities', 'location', 'place', 'destination'],
            'activities': ['activities', 'adventure', 'things to do', 'sports', 'explore', 'entertainment', 'nightlife'],
            'food': ['restaurant', 'cuisine', 'food', 'dining', 'culinary'],
            'planning': ['tips', 'tricks', 'guide', 'planning', 'packing', 'travel']
        }
        
        # First pass: calculate scores and determine categories
        scored_sections = []
        for section in sections:
            score = self._calculate_section_score(section, keywords)
            category = self._determine_category(section, relevant_categories)
            
            # Boost scores for youth-oriented content
            if any(word in section.get("text", "").lower() for word in ['beach', 'nightlife', 'adventure', 'fun', 'party', 'sport']):
                score *= 1.5
                
            scored_sections.append({
                **section,
                "score": score,
                "category": category
            })
        
        # Sort by score within each category
        categorized = defaultdict(list)
        for section in scored_sections:
            categorized[section["category"]].append(section)
        
        for category in categorized:
            categorized[category].sort(key=lambda x: x["score"], reverse=True)
        
        # Construct final ordered list based on priority
        priority_order = ['cities', 'activities', 'food', 'planning']
        final_sections = []
        
        for category in priority_order:
            if category in categorized and categorized[category]:
                final_sections.append(categorized[category][0])
                if category == 'activities' and len(categorized[category]) > 1:
                    final_sections.append(categorized[category][1])
        
        # Fill remaining slots with highest scoring sections from any category
        remaining = sorted(
            [s for s in scored_sections if s not in final_sections],
            key=lambda x: x["score"],
            reverse=True
        )
        
        final_sections.extend(remaining[:max(0, 5 - len(final_sections))])
        return final_sections[:5]
    
    def _calculate_section_score(self, section: Dict, keywords: List[str]) -> float:
        """Calculate relevance score for a section"""
        text = section.get("text", "")
        title = section.get("title", "")
        
        # Base scores
        keyword_score = self._keyword_match_score(text, keywords)
        title_score = self._title_relevance_score(title, keywords)
        
        # Youth-oriented content boost
        youth_keywords = ['beach', 'nightlife', 'adventure', 'fun', 'party', 'sport', 'entertainment']
        youth_score = sum(word.lower() in text.lower() for word in youth_keywords) / len(youth_keywords)
        
        # Group-friendly content boost
        group_keywords = ['group', 'friends', 'together', 'share', 'social', 'activities']
        group_score = sum(word.lower() in text.lower() for word in group_keywords) / len(group_keywords)
        
        # Budget-conscious content boost
        budget_keywords = ['affordable', 'budget', 'cheap', 'deal', 'save', 'cost', 'price']
        budget_score = sum(word.lower() in text.lower() for word in budget_keywords) / len(budget_keywords)
        
        # Optional: semantic similarity if available
        semantic_score = 0.0
        try:
            semantic_score = self.keyword_extractor.compute_semantic_similarity(
                text, " ".join(keywords + youth_keywords + group_keywords + budget_keywords)
            )
        except Exception:
            pass
        
        # Normalize scores between 0 and 1
        weights = {
            'keyword': 0.3,
            'title': 0.2,
            'youth': 0.2,
            'group': 0.15,
            'budget': 0.1,
            'semantic': 0.05
        }
        
        final_score = (
            weights['keyword'] * keyword_score +
            weights['title'] * title_score +
            weights['youth'] * youth_score +
            weights['group'] * group_score +
            weights['budget'] * budget_score +
            weights['semantic'] * semantic_score
        ) / sum(weights.values())
        
        return float(final_score)
    
    def _keyword_match_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword match score"""
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        return matches / (len(keywords) + 1e-6)
    
    def _normalize_length_score(self, length: int) -> float:
        """Normalize content length score"""
        # Prefer medium-length sections (100-500 words)
        if length < 50:
            return 0.3
        elif length < 100:
            return 0.6
        elif length < 500:
            return 1.0
        else:
            return 0.7
    
    def _position_score(self, rel_position: float) -> float:
        """Score based on relative position in document"""
        # Prefer content in the first half of the document
        return 1.0 - (rel_position ** 0.5)
    
    def _title_relevance_score(self, title: str, keywords: List[str]) -> float:
        """Calculate title relevance to keywords"""
        if not title:
            return 0.0
        title_lower = title.lower()
        matches = sum(1 for kw in keywords if kw.lower() in title_lower)
        return matches / (len(keywords) + 1e-6)
