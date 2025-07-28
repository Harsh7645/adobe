import fitz  # PyMuPDF
import re
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

class PDFProcessor:
    def __init__(self):
        self.min_title_font_size = 14
        self.min_heading_font_size = 12

    def extract_text_with_metadata(self, pdf_path: str) -> List[Dict]:
        """Extract text blocks with font, size, and position metadata"""
        doc = fitz.open(pdf_path)
        text_blocks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")

            for block in blocks["blocks"]:
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_blocks.append({
                                "text": span["text"].strip(),
                                "font": span["font"],
                                "size": span["size"],
                                "flags": span["flags"],  # Bold, italic flags
                                "page": page_num + 1,
                                "bbox": span["bbox"],  # Bounding box
                                "line_bbox": line["bbox"]
                            })

        doc.close()
        return text_blocks

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove page numbers and common artifacts
        text = re.sub(r'^\d+\s*$', '', text)
        return text

    def extract_keywords(self, text_blocks: List[Dict], top_n: int = 10) -> list:
        """Extract top N keywords from the document using simple frequency analysis."""
        from collections import Counter
        all_text = ' '.join([self.clean_text(b["text"]) for b in text_blocks if len(b["text"]) > 3])
        words = re.findall(r'\b\w{4,}\b', all_text.lower())  # Only words with 4+ chars
        stopwords = set([
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
        filtered = [w for w in words if w not in stopwords]
        freq = Counter(filtered)
        keywords = [w for w, _ in freq.most_common(top_n)]
        return keywords

    def get_font_statistics(self, text_blocks: List[Dict]) -> Dict:
        """Analyze font usage patterns in the document"""
        font_stats = defaultdict(list)
        size_counts = defaultdict(int)

        for block in text_blocks:
            if len(block["text"]) > 3:  # Ignore very short text
                font_key = f"{block['font']}_{block['size']}"
                font_stats[font_key].append(block)
                size_counts[block["size"]] += 1

        # Find most common body text size
        body_size = max(size_counts.items(), key=lambda x: x[1])[0] if size_counts else 12

        return {
            "font_stats": font_stats,
            "body_size": body_size,
            "size_counts": size_counts
        }
