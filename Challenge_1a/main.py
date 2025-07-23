#!/usr/bin/env python3

import os
import json
import sys
from pathlib import Path
from pdf_processor import PDFProcessor
from heading_detector import HeadingDetector

def process_pdf(pdf_path: str, output_dir: str) -> bool:
    """Process a single PDF and generate JSON outline"""
    try:
        print(f"Processing: {pdf_path}")
        
        # Initialize processors
        pdf_processor = PDFProcessor()
        heading_detector = HeadingDetector()
        
        # Extract text with metadata
        text_blocks = pdf_processor.extract_text_with_metadata(pdf_path)
        if not text_blocks:
            print(f"Warning: No text extracted from {pdf_path}")
            return False
        
        # Analyze font patterns
        font_stats = pdf_processor.get_font_statistics(text_blocks)
        
        # Extract title
        title = heading_detector.extract_title(text_blocks, font_stats)
        
        # Detect headings
        potential_headings = []
        for block in text_blocks:
            is_heading, confidence = heading_detector.is_likely_heading(block, font_stats)
            if is_heading:
                potential_headings.append({
                    **block,
                    "confidence": confidence
                })
        
        # Sort by confidence and position
        potential_headings.sort(key=lambda x: (-x["confidence"], x["page"], x["bbox"][1]))
        
        # Classify heading levels
        outline = []
        for heading in potential_headings:
            level = heading_detector.classify_heading_level(heading, potential_headings, font_stats)
            
            outline.append({
                "level": level,
                "text": pdf_processor.clean_text(heading["text"]),
                "page": heading["page"]
            })
        
        # Remove duplicates and very similar headings
        outline = remove_duplicate_headings(outline)
        
        # Create output JSON
        result = {
            "title": title,
            "outline": outline
        }
        
        # Write output file
        pdf_name = Path(pdf_path).stem
        output_path = os.path.join(output_dir, f"{pdf_name}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Generated: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return False

def remove_duplicate_headings(outline: list) -> list:
    """Remove duplicate or very similar headings"""
    seen = set()
    filtered_outline = []
    
    for item in outline:
        # Create a normalized version for comparison
        normalized = item["text"].lower().strip()
        key = (normalized, item["page"])
        
        if key not in seen and len(normalized) > 2:
            seen.add(key)
            filtered_outline.append(item)
    
    return filtered_outline

def main():
    """Main processing function"""
    input_dir = "./sample_dataset/pdfs"
    output_dir = "./sample_dataset/preds"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files
    pdf_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(input_dir, file))
    
    if not pdf_files:
        print("No PDF files found in input directory")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    success_count = 0
    for pdf_path in pdf_files:
        if process_pdf(pdf_path, output_dir):
            success_count += 1
    
    print(f"Successfully processed {success_count}/{len(pdf_files)} PDFs")

if __name__ == "__main__":
    main()
