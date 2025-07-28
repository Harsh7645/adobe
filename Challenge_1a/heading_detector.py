import re
from typing import List, Dict, Optional, Tuple
from collections import Counter

import joblib
import numpy as np

class HeadingDetector:
    def __init__(self, model_path: str = 'heading_classifier.pkl'):
        # Common heading patterns
        self.heading_patterns = [
            r'^(chapter|section|part)\s+\d+',
            r'^\d+\.\s+',  # 1. Introduction
            r'^\d+\.\d+\s+',  # 1.1 Overview
            r'^\d+\.\d+\.\d+\s+',  # 1.1.1 Details
            r'^[A-Z][A-Z\s]{10,}$',  # ALL CAPS headings
            r'^(introduction|conclusion|summary|abstract|methodology|results|discussion|references)$'
        ]
        # Load ML model if available
        try:
            self.model = joblib.load(model_path)
            print(f"[INFO][HeadingDetector] Loaded ML heading classifier from {model_path}")
        except Exception as e:
            self.model = None
            print(f"[WARNING][HeadingDetector] Could not load ML model: {e}")

    def extract_features(self, block: Dict, font_stats: Dict) -> np.ndarray:
        """Extract enhanced features for ML model from a text block (20+ features)"""
        text = block["text"].strip()
        size = block["size"]
        bbox = block.get("bbox", [0, 0, 0, 0])
        page = block.get("page", 1)
        num_words = len(text.split())
        num_chars = len(text)
        is_all_caps = int(text.isupper())
        y0 = bbox[1]
        y1 = bbox[3]
        rel_font_size = size / (font_stats["body_size"] + 1e-6)
        has_numbering = int(bool(re.match(r"^(\d+\.|\d+\.\d+|[A-Z][a-z]+) ", text)))
        avg_word_len = np.mean([len(w) for w in text.split()]) if num_words > 0 else 0
        num_digits = sum(c.isdigit() for c in text)
        num_upper = sum(c.isupper() for c in text)
        num_lower = sum(c.islower() for c in text)
        pct_digits = num_digits / (num_chars + 1e-6)
        pct_upper = num_upper / (num_chars + 1e-6)
        pct_lower = num_lower / (num_chars + 1e-6)
        ends_with_colon = int(text.endswith(':'))
        starts_with_bullet = int(text.lstrip().startswith(('-', '*', '\u2022')))
        # New features
        font_weight = block.get("font_weight", 0)  # 0=normal, 1=bold (if available)
        font_name = block.get("font_name", "")
        is_bold = int("bold" in font_name.lower() or font_weight == 1)
        is_italic = int("italic" in font_name.lower())
        is_centered = int(abs((bbox[0] + bbox[2]) / 2 - font_stats.get("page_width", 595) / 2) < font_stats.get("page_width", 595) * 0.15)
        whitespace_above = block.get("whitespace_above", 0)
        whitespace_below = block.get("whitespace_below", 0)
        whitespace_ratio = (whitespace_above + whitespace_below) / (font_stats.get("page_height", 800) + 1e-6)
        rel_y0 = y0 / (font_stats.get("page_height", 800) + 1e-6)
        rel_y1 = y1 / (font_stats.get("page_height", 800) + 1e-6)
        # 20+ features total
        return np.array([
            size,              # 1
            num_words,         # 2
            num_chars,         # 3
            is_all_caps,       # 4
            y0,                # 5
            y1,                # 6
            rel_font_size,     # 7
            has_numbering,     # 8
            avg_word_len,      # 9
            num_digits,        #10
            num_upper,         #11
            num_lower,         #12
            pct_digits,        #13
            pct_upper,         #14
            pct_lower,         #15
            ends_with_colon,   #16
            starts_with_bullet,#17
            is_bold,           #18
            is_italic,         #19
            is_centered,       #20
            whitespace_ratio,  #21
            rel_y0,            #22
            rel_y1             #23
        ])

    def predict_heading_ml(self, block: Dict, font_stats: Dict) -> Tuple[bool, float]:
        """Predict heading using ML model. Returns (is_heading, confidence)"""
        if self.model is None:
            return False, 0.0
        features = self.extract_features(block, font_stats).reshape(1, -1)
        pred = self.model.predict(features)[0]
        if hasattr(self.model, "predict_proba"):
            conf = float(np.max(self.model.predict_proba(features)))
        else:
            conf = 1.0 if pred else 0.0
        return bool(pred), conf
    
    def is_likely_heading(self, block: Dict, font_stats: Dict) -> Tuple[bool, float]:
        """Hybrid heading detection: use ML model if available, else fallback to rules"""
        # Try ML model first
        if self.model is not None:
            is_heading, conf = self.predict_heading_ml(block, font_stats)
            print(f"[DEBUG][is_likely_heading][ML] Text: '{block['text'][:60]}' | Pred: {is_heading} | Conf: {conf:.2f}")
            return is_heading, conf
        # Fallback to rule-based
        size = block["size"]
        text = block["text"].strip()
        page = block.get("page", 1)
        bbox = block.get("bbox", [0, 0, 0, 0])
        print(f"[DEBUG][is_likely_heading][RULE] Text: '{text}' | Size: {size} | Page: {page} | BBox: {bbox}")
        reason = None
        blacklist = {"page", "table of contents", "contents", "index", "copyright", "revision history", "overview"}
        if text.lower() in blacklist:
            print("[DEBUG][is_likely_heading] REJECTED: Blacklist")
            return False, 0.0
        if len(text) < 5:
            print("[DEBUG][is_likely_heading] REJECTED: Too short")
            return False, 0.0
        if text.isdigit():
            print("[DEBUG][is_likely_heading] REJECTED: Is digit")
            return False, 0.0
        if bbox[1] < 30 or bbox[3] > 750:
            print("[DEBUG][is_likely_heading] REJECTED: Header/Footer position")
            return False, 0.0
        if len(text.split()) < 2:
            print("[DEBUG][is_likely_heading] REJECTED: Too few words")
            return False, 0.0
        if size < font_stats["body_size"] * 1.12:
            print(f"[DEBUG][is_likely_heading] REJECTED: Font size too small (body_size={font_stats['body_size']})")
            return False, 0.0
        if re.match(r"^(\d+\.|\d+\.\d+|[A-Z][a-z]+) ", text):
            print("[DEBUG][is_likely_heading] ACCEPTED: Numbered/Keyword heading")
            return True, 1.0
        keywords = ["introduction", "summary", "background", "references", "acknowledgements", "objectives", "requirements", "content", "audience", "trademarks", "web sites"]
        if any(kw in text.lower() for kw in keywords):
            print("[DEBUG][is_likely_heading] ACCEPTED: Section keyword")
            return True, 0.95
        if size >= font_stats["body_size"] * 1.25:
            print("[DEBUG][is_likely_heading] ACCEPTED: Large font size")
            return True, 0.85
        print("[DEBUG][is_likely_heading] REJECTED: Did not meet any criteria")
        return False, 0.0
    
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


        # Restrict to blocks in the top 40% of the page (assuming bbox[1] is y0, bbox[3] is y1)
        if first_page_blocks:
            page_height = max(b["bbox"][3] for b in first_page_blocks)
        else:
            page_height = 800  # fallback
        top_cutoff = page_height * 0.40
        threshold = max_size * 0.85

        # Filter: only keep blocks that look like real title lines
        def is_good_title_line(text):
            text = text.strip()
            if len(text) < 6:
                return False
            if text.lower() in ["page", "table of contents", "contents", "index"]:
                return False
            if not text[0].isupper():
                return False
            if text.isdigit():
                return False
            return True

        seen_texts = set()
        top_blocks = [b for b in first_page_blocks if b["size"] >= threshold and len(b["text"]) > 3 and b["bbox"][1] < top_cutoff and is_good_title_line(b["text"]) and b["text"].strip().lower() not in seen_texts and not seen_texts.add(b["text"].strip().lower())]
        sorted_blocks = sorted(top_blocks, key=lambda x: x["bbox"][1])

        # Only concatenate the first group of consecutive blocks (stop at first vertical gap > 40)
        group = []
        last_y = None
        for block in sorted_blocks:
            if last_y is None or abs(block["bbox"][1] - last_y) < 40:
                group.append(block)
            else:
                break
            last_y = block["bbox"][1]

        if group:
            title_text = ' '.join(self._clean_title(b["text"]) for b in group)
            print("[DEBUG][extract_title] Multi-line title candidates:")
            for b in group:
                print(f"  - '{b['text'][:60]}' | size: {b['size']} | bbox: {b['bbox']}")
            print(f"[DEBUG][extract_title] Selected multi-line title: '{title_text}'")
            return title_text

        # Fallback: use the single largest text block on the first page
        print("[DEBUG][extract_title] No suitable group found, falling back to largest block.")
        large_blocks = [b for b in first_page_blocks if b["size"] == max_size and len(b["text"]) > 3]
        if large_blocks:
            block = min(large_blocks, key=lambda x: x["bbox"][1])
            print(f"[DEBUG][extract_title] Fallback block: '{block['text']}' | size: {block['size']} | bbox: {block['bbox']}")
            return self._clean_title(block["text"])

        print("[DEBUG][extract_title] No suitable title found, returning 'Untitled Document'")
        return "Untitled Document"
    
    def _clean_title(self, title: str) -> str:
        """Clean extracted title"""
        title = re.sub(r'\s+', ' ', title).strip()
        # Remove common title artifacts
        title = re.sub(r'^(title|document):\s*', '', title, flags=re.IGNORECASE)
        return title
