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
        body_size = max(size_counts.items(), key=lambda x: x[1])[0]
        
        return {
            "font_stats": font_stats,
            "body_size": body_size,
            "size_counts": size_counts
        }
