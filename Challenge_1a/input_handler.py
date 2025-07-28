import json
from pathlib import Path

def load_persona(persona_json_path):
    with open(persona_json_path, 'r', encoding='utf-8') as f:
        persona = json.load(f)
    return persona

def list_pdf_files(pdf_dir):
    pdf_dir = Path(pdf_dir)
    return sorted([str(f) for f in pdf_dir.glob('*.pdf')])
