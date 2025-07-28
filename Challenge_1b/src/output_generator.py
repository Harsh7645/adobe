from typing import Dict, List
import json
from datetime import datetime

class OutputGenerator:
    def __init__(self):
        pass
        
    def generate_output(self, 
                       ranked_sections: List[Dict],
                       metadata: Dict,
                       input_data: Dict,
                       pdf_files: List[str]) -> Dict:
        """Generate structured output JSON"""
        output = {
            "metadata": {
                "input_documents": [pdf.split('/')[-1].split('\\')[-1] for pdf in pdf_files],
                "persona": input_data.get("persona", {}).get("role", ""),
                "job_to_be_done": input_data.get("job_to_be_done", {}).get("task", ""),
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
        
        # Process ranked sections for main structure
        for i, section in enumerate(ranked_sections):
            processed_section = {
                "document": section.get("source_file", "").split('/')[-1].split('\\')[-1],
                "section_title": section.get("title", "Untitled Section"),
                "importance_rank": i + 1,
                "page_number": section.get("page", 1)
            }
            output["extracted_sections"].append(processed_section)
            
            # Add detailed subsection analysis
            subsection = {
                "document": section.get("source_file", "").split('/')[-1].split('\\')[-1],
                "refined_text": section.get("text", ""),
                "page_number": section.get("page", 1)
            }
            output["subsection_analysis"].append(subsection)
        
        return output
    
    def save_output(self, output: Dict, output_path: str):
        """Save output to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
