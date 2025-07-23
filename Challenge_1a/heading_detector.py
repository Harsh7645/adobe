import re
from typing import List, Dict, Optional, Tuple
from collections import Counter

class HeadingDetector:
    def __init__(self):
        # Common heading patterns
        self.heading_patterns = [
            r'^(chapter|section|part)\s+\d+',
            r'^\d+\.\s+',  # 1. Introduction
            r'^\d+\.\d+\s+',  # 1.1 Overview
            r'^\d+\.\d+\.\d+\s+',  # 1.1.1 Details
            r'^[A-Z][A-Z\s]{10,}$',  # ALL CAPS headings
            r'^(introduction|conclusion|summary|abstract|methodology|results|discussion|references)$'
        ]
    
    def is_likely_heading(self, block: Dict, font_stats: Dict) -> Tuple[bool, float]:
        """Determine if a text block is likely a heading with confidence score"""
        text = block["text"].strip()
        
        if len(text) < 3 or len(text) > 200:  # Too short or too long
            return False, 0.0
        
        confidence = 0.0
        
        # Font size analysis
        body_size = font_stats["body_size"]
        if block["size"] > body_size * 1.2:  # Significantly larger than body
            confidence += 0.4
        elif block["size"] > body_size * 1.1:  # Slightly larger than body
            confidence += 0.2
        
        # Font weight (bold)
        if block["flags"] & 16:  # Bold flag
            confidence += 0.3
        
        # Position analysis - headings often start near left margin
        left_margin = block["bbox"][0]
        if left_margin < 100:  # Near left margin
            confidence += 0.1
        
        # Pattern matching
        text_lower = text.lower()
        for pattern in self.heading_patterns:
            if re.search(pattern, text_lower):
                confidence += 0.3
                break
        
        # All caps (but not too long)
        if text.isupper() and 5 <= len(text) <= 50:
            confidence += 0.2
        
        # Standalone line (not part of paragraph)
        if self._is_standalone_line(block):
            confidence += 0.2
        
        # Common heading words
        heading_keywords = ["introduction", "background", "methodology", "results", 
                          "conclusion", "summary", "overview", "abstract"]
        if any(keyword in text_lower for keyword in heading_keywords):
            confidence += 0.2
        
        return confidence > 0.5, confidence
    
    def _is_standalone_line(self, block: Dict) -> bool:
        """Check if text appears to be on its own line"""
        line_height = block["line_bbox"][3] - block["line_bbox"][1]
        text_height = block["bbox"][3] - block["bbox"][1]
        
        # If text takes up most of the line height, likely standalone
        return text_height / line_height > 0.8
    
    def classify_heading_level(self, block: Dict, all_headings: List[Dict], font_stats: Dict) -> str:
        """Classify heading into H1, H2, or H3 based on context"""
        text = block["text"].strip()
        size = block["size"]
        
        # Get sizes of all detected headings
        heading_sizes = [h["size"] for h in all_headings]
        unique_sizes = sorted(set(heading_sizes), reverse=True)
        
        # Pattern-based classification
        if re.search(r'^(chapter|part)\s+\d+', text.lower()):
            return "H1"
        elif re.search(r'^\d+\.\s+', text):
            return "H1"
        elif re.search(r'^\d+\.\d+\s+', text):
            return "H2"
        elif re.search(r'^\d+\.\d+\.\d+\s+', text):
            return "H3"
        
        # Size-based classification
        if len(unique_sizes) >= 3:
            if size >= unique_sizes[0]:
                return "H1"
            elif size >= unique_sizes[1]:
                return "H2"
            else:
                return "H3"
        elif len(unique_sizes) == 2:
            return "H1" if size >= unique_sizes[0] else "H2"
        else:
            return "H1"
    
    def extract_title(self, text_blocks: List[Dict], font_stats: Dict) -> Optional[str]:
        """Extract document title from first page"""
        first_page_blocks = [b for b in text_blocks if b["page"] == 1]
        
        # Look for largest text on first page
        candidates = []
        max_size = max(b["size"] for b in first_page_blocks if len(b["text"]) > 5)
        
        for block in first_page_blocks:
            if (block["size"] >= max_size * 0.9 and 
                5 <= len(block["text"]) <= 200 and
                not re.search(r'^\d+$', block["text"])):  # Not just a number
                candidates.append(block)
        
        if candidates:
            # Choose the one highest on the page
            title_block = min(candidates, key=lambda x: x["bbox"][1])
            return self._clean_title(title_block["text"])
        
        return "Untitled Document"
    
    def _clean_title(self, title: str) -> str:
        """Clean extracted title"""
        title = re.sub(r'\s+', ' ', title).strip()
        # Remove common title artifacts
        title = re.sub(r'^(title|document):\s*', '', title, flags=re.IGNORECASE)
        return title
