import os
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# --- Feature extraction for each text block ---
def extract_features(block, font_stats):
    text = block["text"].strip()
    size = block["size"]
    bbox = block.get("bbox", [0, 0, 0, 0])
    page = block.get("page", 1)
    num_words = len(text.split())
    num_chars = len(text)
    is_all_caps = int(text.isupper())
    y0 = bbox[1]
    y1 = bbox[3]
    rel_font_size = size / (font_stats["body_size"] + 1e-6) if font_stats["body_size"] else 1.0
    has_numbering = int(bool(re.match(r"^(\d+\.|\d+\.\d+|[A-Z][a-z]+) ", text)))
    avg_word_len = np.mean([len(w) for w in text.split()]) if num_words > 0 else 0
    num_digits = sum(c.isdigit() for c in text)
    num_upper = sum(c.isupper() for c in text)
    num_lower = sum(c.islower() for c in text)
    pct_digits = num_digits / (num_chars + 1e-6)
    pct_upper = num_upper / (num_chars + 1e-6)
    pct_lower = num_lower / (num_chars + 1e-6)
    ends_with_colon = int(text.endswith(':'))
    starts_with_bullet = int(text.lstrip().startswith(('-', '*', '\u2022')))
    font_weight = block.get("font_weight", 0)  # 0=normal, 1=bold (if available)
    font_name = block.get("font_name", "")
    is_bold = int("bold" in font_name.lower() or font_weight == 1)
    is_italic = int("italic" in font_name.lower())
    is_centered = int(abs((bbox[0] + bbox[2]) / 2 - font_stats.get("page_width", 595) / 2) < font_stats.get("page_width", 595) * 0.15)
    whitespace_above = block.get("whitespace_above", 0)
    whitespace_below = block.get("whitespace_below", 0)
    whitespace_ratio = (whitespace_above + whitespace_below) / (font_stats.get("page_height", 800) + 1e-6)
    rel_y0 = y0 / (font_stats.get("page_height", 800) + 1e-6)
    rel_y1 = y1 / (font_stats.get("page_height", 800) + 1e-6)
    # 23 features total
    feats = np.array([
        size,              # 1
        num_words,         # 2
        num_chars,         # 3
        is_all_caps,       # 4
        y0,                # 5
        y1,                # 6
        rel_font_size,     # 7
        has_numbering,     # 8
        avg_word_len,      # 9
        num_digits,        #10
        num_upper,         #11
        num_lower,         #12
        pct_digits,        #13
        pct_upper,         #14
        pct_lower,         #15
        ends_with_colon,   #16
        starts_with_bullet,#17
        is_bold,           #18
        is_italic,         #19
        is_centered,       #20
        whitespace_ratio,  #21
        rel_y0,            #22
        rel_y1             #23
    ])
    return feats, {
        "size": size,
        "num_words": num_words,
        "num_chars": num_chars,
        "is_all_caps": is_all_caps,
        "y0": y0,
        "y1": y1,
        "rel_font_size": rel_font_size,
        "has_numbering": has_numbering,
        "avg_word_len": avg_word_len,
        "num_digits": num_digits,
        "num_upper": num_upper,
        "num_lower": num_lower,
        "pct_digits": pct_digits,
        "pct_upper": pct_upper,
        "pct_lower": pct_lower,
        "ends_with_colon": ends_with_colon,
        "starts_with_bullet": starts_with_bullet,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "is_centered": is_centered,
        "whitespace_ratio": whitespace_ratio,
        "rel_y0": rel_y0,
        "rel_y1": rel_y1
    }

# --- Labeling: align block with expected headings ---
def is_heading_label(block, expected_headings):
    text = block["text"].strip().lower()
    page = block.get("page", 1)
    for h in expected_headings:
        if h["text"].strip().lower() == text and h["page"] == page:
            return 1
    return 0

# --- Data preparation ---
def prepare_training_data(pdf_dir, expected_dir, pdf_processor, font_stats_func):
    X, y, feature_dicts = [], [], []
    for pdf_file in os.listdir(pdf_dir):
        if not pdf_file.lower().endswith(".pdf"): continue
        pdf_path = os.path.join(pdf_dir, pdf_file)
        json_name = Path(pdf_file).stem + ".json"
        expected_path = os.path.join(expected_dir, json_name)
        if not os.path.exists(expected_path): continue
        # Extract blocks and font stats
        blocks = pdf_processor.extract_text_with_metadata(pdf_path)
        font_stats = font_stats_func(blocks)
        # Load expected headings
        with open(expected_path, "r", encoding="utf-8") as f:
            expected = json.load(f)
        expected_headings = expected.get("outline", [])
        # For each block, extract features and label
        for block in blocks:
            feats, feat_dict = extract_features(block, font_stats)
            label = is_heading_label(block, expected_headings)
            X.append(feats)
            y.append(label)
            feature_dicts.append(feat_dict)
    return np.array(X), np.array(y), feature_dicts

# --- Model training ---
def train_heading_classifier(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return clf

# --- Save model ---
def save_model(clf, path="heading_classifier.pkl"):
    joblib.dump(clf, path)
    print(f"Model saved to {path}")

# --- Example usage (to be run as a script) ---
if __name__ == "__main__":
    # Import your PDF processor and font stats function
    from pdf_processor import PDFProcessor
    pdf_processor = PDFProcessor()
    from pdf_processor import PDFProcessor
    font_stats_func = PDFProcessor().get_font_statistics
    pdf_dir = "./sample_dataset/pdfs"
    expected_dir = "./sample_dataset/expected"
    X, y, feature_dicts = prepare_training_data(pdf_dir, expected_dir, pdf_processor, font_stats_func)
    print(f"Extracted {len(X)} samples for training.")
    clf = train_heading_classifier(X, y)
    save_model(clf)
