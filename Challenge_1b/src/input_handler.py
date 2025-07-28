import json
import os
from typing import Dict, List
from pathlib import Path

class InputHandler:
    def __init__(self, input_json_path: str, pdf_dir: str):
        self.input_json_path = input_json_path
        self.pdf_dir = pdf_dir
        
    def load_input_json(self) -> Dict:
        """Load and validate the input JSON containing persona and task details"""
        with open(self.input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Validate required fields
        if 'persona' not in data or 'role' not in data['persona']:
            raise ValueError("Missing required field: role in persona")
        if 'job_to_be_done' not in data or 'task' not in data['job_to_be_done']:
            raise ValueError("Missing required field: task in job_to_be_done")
                
        return data
    
    def get_pdf_files(self) -> List[str]:
        """Get list of PDF files from the specified directory"""
        pdf_files = []
        for file in os.listdir(self.pdf_dir):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(self.pdf_dir, file))
        return pdf_files

    def extract_task_keywords(self, data: Dict) -> List[str]:
        """Extract initial keywords from the task description"""
        task_text = data.get('job_to_be_done', {}).get('task', '').lower()
        # Basic keyword extraction (to be enhanced)
        keywords = [word.strip() for word in task_text.split()]
        return keywords
