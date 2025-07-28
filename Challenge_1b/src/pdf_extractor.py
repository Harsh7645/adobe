import fitz  # PyMuPDF
from typing import Dict, List, Tuple
import re

class PDFExtractor:
    def __init__(self):
        self.font_size_threshold = 12  # Default threshold for heading detection
        
    def extract_text_with_metadata(self, pdf_path: str) -> List[Dict]:
        """Extract text blocks with metadata from PDF"""
        doc = fitz.open(pdf_path)
        blocks = []
        
        for page_num, page in enumerate(doc):
            # Get page dimensions for relative positioning
            page_width = page.rect.width
            page_height = page.rect.height
            
            # Extract blocks with detailed formatting info
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                            
                        blocks.append({
                            "text": text,
                            "page": page_num + 1,
                            "bbox": span["bbox"],
                            "font_size": span["size"],
                            "font_name": span["font"],
                            "is_bold": "bold" in span["font"].lower(),
                            "color": span.get("color", 0),
                            "rel_height": span["bbox"][3] / page_height,
                            "rel_width": (span["bbox"][2] - span["bbox"][0]) / page_width
                        })
        
        return blocks
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
        
    def blocks_to_sections(self, blocks: List[Dict], source_file: str) -> List[Dict]:
        """Convert text blocks into meaningful sections"""
        current_section = None
        sections = []
        
        for block in blocks:
            text = block.get("text", "").strip()
            font_size = block.get("font_size", 0)
            is_bold = block.get("is_bold", False)
            
            # Potential heading detection
            is_heading = (font_size > self.font_size_threshold and len(text) < 100) or is_bold
            
            if is_heading:
                # Save previous section if exists
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "title": text,
                    "text": "",
                    "page": block.get("page", 1),
                    "source_file": source_file
                }
            elif current_section:
                # Add to current section
                current_section["text"] += " " + text
            else:
                # No section started yet, create one without title
                current_section = {
                    "title": "Introduction",
                    "text": text,
                    "page": block.get("page", 1),
                    "source_file": source_file
                }
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        # Clean section text
        for section in sections:
            section["text"] = self.clean_text(section["text"])
            
        return sections
    
    def get_document_metadata(self, pdf_path: str) -> Dict:
        """Extract document metadata"""
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        doc.close()
        return metadata
