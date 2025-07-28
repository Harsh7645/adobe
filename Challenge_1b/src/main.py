#!/usr/bin/env python3

import os
import json
import sys
from pathlib import Path
from input_handler import InputHandler
from pdf_extractor import PDFExtractor
from keyword_extractor import KeywordExtractor
from section_ranker import SectionRanker
from output_generator import OutputGenerator

def main():
    # Initialize components
    input_json = "./input/challenge1b_input.json"
    pdf_dir = "./input/PDFs"
    output_path = "./output/challenge1b_output.json"
    
    # Create input handler
    input_handler = InputHandler(input_json, pdf_dir)
    
    try:
        # Load input data
        input_data = input_handler.load_input_json()
        pdf_files = input_handler.get_pdf_files()
        
        if not pdf_files:
            print("No PDF files found in input directory")
            sys.exit(1)
            
        # Initialize other components
        pdf_extractor = PDFExtractor()
        keyword_extractor = KeywordExtractor()
        section_ranker = SectionRanker(keyword_extractor)
        output_generator = OutputGenerator()
        
        # Process each PDF and collect sections
        all_sections = []
        task_keywords = input_handler.extract_task_keywords(input_data)
        enhanced_keywords = keyword_extractor.add_domain_keywords(
            task_keywords,
            input_data.get("domain", "technical")
        )
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file}")
            
            # Extract text and metadata
            blocks = pdf_extractor.extract_text_with_metadata(pdf_file)
            sections = pdf_extractor.blocks_to_sections(blocks, pdf_file)
            
            # Add section relevance scores
            for section in sections:
                section["relevance_score"] = keyword_extractor.calculate_semantic_similarity(
                    section["text"],
                    " ".join(enhanced_keywords)
                )
            
            all_sections.extend(sections)
        
        # Rank the sections
        ranked_sections = section_ranker.rank_sections(all_sections, enhanced_keywords)
        
        # Generate final output
        output = output_generator.generate_output(
            ranked_sections[:5],  # Only take top 5 sections as per expected format
            {},  # No metadata needed with new format
            input_data,
            pdf_files
        )
        
        # Save output
        output_generator.save_output(output, output_path)
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
