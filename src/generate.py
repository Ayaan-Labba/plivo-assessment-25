import json
import re
import random
import math
from faker import Faker
from typing import List, Dict, Any

# Initialize Faker
fake = Faker()

# --- CONFIGURATION ---
INPUT_FILE = 'data/raw.jsonl' # Save your provided text into this file first
OUTPUT_DIR = 'data'
RANDOM_SEED = 42

# --- NOISE FUNCTIONS (STT Simulation) ---
def noise_digits(text: str) -> str:
    """
    Converts "1234" -> "one 2 three 4" or "1 2 3 4"
    """
    digit_map = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    noisy_text = []
    for char in text:
        if char.isdigit():
            # 50% chance to convert digit to word, otherwise keep digit but ensure spacing
            if random.random() > 0.5:
                noisy_text.append(digit_map[char])
            else:
                noisy_text.append(char)
        else:
            noisy_text.append(char)
    return " ".join(noisy_text)

def noise_email(text: str) -> str:
    """Converts 'john@gmail.com' -> 'john at gmail dot com'"""
    if random.random() > 0.5:
        text = text.replace("@", " at ")
    
    if random.random() > 0.5:
        text = text.replace(".", " dot ")
    
    if random.random() > 0.5:
        text = text.lower()

    return text

# --- ENTITY GENERATORS ---
def get_entity_value(label: str) -> str:
    """Generates a noisy value based on the placeholder label."""
    label = label.strip()
    
    if label == "CREDIT_CARD":
        return noise_digits(fake.credit_card_number())
    
    elif label == "PHONE":
        # Strip formatting first, then noise
        raw = "".join([c for c in fake.phone_number() if c.isdigit()])
        return noise_digits(raw)
    
    elif label == "EMAIL":
        return noise_email(fake.email())
    
    elif label == "PERSON_NAME":
        return fake.name().lower()
    
    elif label == "DATE":
        # Spoken date: "January first 2024"
        return fake.date_object().strftime("%B %d %Y").lower()
    
    elif label == "CITY":
        return fake.city().lower()
    
    elif label == "LOCATION":
        # Simplified address
        addr = fake.address().replace("\n", " ").lower()
        return " ".join(addr.split()[:3])
        
    else:
        return "unknown_entity"

# --- CORE LOGIC ---
def process_text(template: str, utt_id: str) -> Dict[str, Any]:
    """
    Parses a template string, replaces {{PLACEHOLDERS}}, 
    and calculates character offsets.
    """
    # Regex to find {{ENTITY}}
    pattern = r'(\{\{[A-Z_]+\}\})'
    parts = re.split(pattern, template)
    
    final_text = ""
    entities = []
    cursor = 0
    
    for part in parts:
        # Check if part is a placeholder (e.g., "{{CREDIT_CARD}}")
        if part.startswith("{{") and part.endswith("}}"):
            label = part[2:-2] # Remove brackets
            
            # Generate noisy value
            value = get_entity_value(label)
            
            # Record Entity Span
            start = cursor
            end = cursor + len(value)
            
            entities.append({
                "start": start,
                "end": end,
                "label": label
            })
            
            # Update text and cursor
            final_text += value
            cursor += len(value)
            
        else:
            # It's just normal text
            text_part = part.lower() # Lowercase normal text for consistency
            final_text += text_part
            cursor += len(text_part)
            
    return {
        "id": utt_id,
        "text": final_text,
        "entities": entities
    }

def main():
    random.seed(RANDOM_SEED)
    Faker.seed(RANDOM_SEED)
    
    # 1. Load Raw Data
    try:
        with open(INPUT_FILE, 'r') as f:
            raw_data = json.load(f)
            templates = raw_data['text']
    except FileNotFoundError:
        print(f"Error: Please create '{INPUT_FILE}' with your data first.")
        return

    print(f"Loaded {len(templates)} templates.")

    # 2. Process All Lines
    processed_data = []
    for i, template in enumerate(templates):
        utt_id = f"utt_{i:04d}"
        entry = process_text(template, utt_id)
        processed_data.append(entry)

    # 3. Shuffle and Split (8:1:1)
    random.shuffle(processed_data)
    
    total = len(processed_data)
    n_train = int(total * 0.8)
    n_dev = int(total * 0.1)
    # Test gets the remainder to ensure no data loss
    
    train_set = processed_data[:n_train]
    dev_set = processed_data[n_train : n_train + n_dev]
    test_set = processed_data[n_train + n_dev:]
    
    # 4. Save to JSONL
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def save_jsonl(data, filename):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved {len(data)} examples to {path}")

    save_jsonl(train_set, 'train_new.jsonl')
    save_jsonl(dev_set, 'dev_new.jsonl')
    
    # Test set typically doesn't need labels for the assignment inference step, 
    # but keeping them helps you validate. The assignment asks for test.jsonl (no labels)
    # but we will save with labels for your own reference.
    save_jsonl(test_set, 'test_new.jsonl')

if __name__ == "__main__":
    main()