import re
from typing import List, Dict

def detect_titles(text_blocks: List[Dict]) -> List[Dict]:
    # Detect possible section titles using font size, bold, and regex patterns
    titles = []
    for block in text_blocks:
        text = block['text'].strip()
        if len(text) < 3 or len(text) > 120:
            continue
        # Heuristic: large font or bold or matches heading pattern
        if block['size'] > 13 or (block['flags'] & 16):
            if re.match(r'^(\d+\.|[A-Z][A-Z\s]{4,}|[A-Z][a-z]+( [A-Z][a-z]+)*:?)', text):
                titles.append(block)
    return titles
