import fitz
from typing import List, Dict

class PDFExtractor:
    def extract_text_blocks(self, pdf_path: str) -> List[Dict]:
        doc = fitz.open(pdf_path)
        text_blocks = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_blocks.append({
                                "text": span["text"].strip(),
                                "font": span["font"],
                                "size": span["size"],
                                "flags": span["flags"],
                                "page": page_num + 1,
                                "bbox": span["bbox"],
                                "line_bbox": line["bbox"]
                            })
        doc.close()
        return text_blocks
