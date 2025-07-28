import json
from datetime import datetime

def write_output(output_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

def build_output_json(persona, input_docs, sections):
    return {
        "metadata": {
            "input_docs": input_docs,
            "persona": persona.get("persona"),
            "role": persona.get("role"),
            "job_to_be_done": persona.get("job_to_be_done"),
            "timestamp": datetime.now().isoformat()
        },
        "sections": sections
    }
