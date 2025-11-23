import json
import random
from faker import Faker
from typing import List, Dict, Tuple

fake = Faker()

# Configuration
OUTPUT_DIR = "data"
NUM_TRAIN = 800  # Target: 500-1000
NUM_DEV = 200    # Target: 100-200

# Entity categories as per assignment [cite: 8]
ENTITIES = [
    "CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", 
    "DATE", "CITY", "LOCATION"
]

# Mapping digits to spoken words for STT noise
DIGIT_MAP = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}

def noise_digits(text: str) -> str:
    """
    Simulates STT by spacing out digits or converting some to words.
    E.g., "1234" -> "1 2 3 4" or "one two three four"
    """
    noisy_text = []
    for char in text:
        if char.isdigit():
            if random.random() > 0.5:
                noisy_text.append(DIGIT_MAP[char])
            else:
                noisy_text.append(char) # Keep digit but it will be spaced later
        else:
            noisy_text.append(char)
    
    # Join with spaces to simulate pausing/tokenization in speech
    return " ".join(noisy_text)

def noise_email(text: str) -> str:
    """
    Simulates STT email reading.
    E.g., "john@gmail.com" -> "john at gmail dot com"
    """
    text = text.replace("@", " at ").replace(".", " dot ")
    return text.lower()

def generate_example(id_counter: int) -> Dict:
    entity_type = random.choice(ENTITIES)
    
    # 1. Generate raw entity and template
    if entity_type == "CREDIT_CARD":
        raw_val = fake.credit_card_number()
        val = noise_digits(raw_val)
        templates = [
            "my credit card is {val}",
            "charge it to {val} please",
            "card number {val}"
        ]
        
    elif entity_type == "PHONE":
        raw_val = fake.phone_number()
        # Strip formatting like ( ) - before noising
        clean_val = "".join([c for c in raw_val if c.isdigit()])
        val = noise_digits(clean_val)
        templates = [
            "call me at {val}",
            "my number is {val}",
            "you can reach me on {val}"
        ]
        
    elif entity_type == "EMAIL":
        raw_val = fake.email()
        val = noise_email(raw_val)
        templates = [
            "my email is {val}",
            "send it to {val}",
            "contact {val} for details"
        ]
        
    elif entity_type == "PERSON_NAME":
        val = fake.name().lower()
        templates = [
            "my name is {val}",
            "this is {val} speaking",
            "ask for {val}"
        ]
        
    elif entity_type == "DATE":
        # STT dates are rarely ISO format. They are spoken.
        date_obj = fake.date_object()
        # Example: "january first twenty twenty three"
        val = date_obj.strftime("%B %d %Y").lower() 
        templates = [
            "the date is {val}",
            "i was born on {val}",
            "schedule it for {val}"
        ]
        
    elif entity_type == "CITY":
        val = fake.city().lower()
        templates = [
            "i live in {val}",
            "heading to {val}",
            "how is the weather in {val}"
        ]
        
    elif entity_type == "LOCATION":
        val = fake.address().replace("\n", " ").lower()
        # Simplify address for STT (remove zip codes often ignored in speech context or separate entity)
        val = " ".join(val.split()[:3]) 
        templates = [
            "meet me at {val}",
            "location is {val}",
            "deliver to {val}"
        ]

    # 2. Construct the sentence
    template = random.choice(templates)
    full_text = template.format(val=val)
    
    # 3. Find the indices (character offsets)
    # Note: This finds the first occurrence. In complex data, you'd need robust index tracking.
    start_index = full_text.find(val)
    end_index = start_index + len(val)
    
    return {
        "id": f"utt_{id_counter:04d}",
        "text": full_text,
        "entities": [
            {
                "start": start_index,
                "end": end_index,
                "label": entity_type
            }
        ]
    }

def save_jsonl(data: List[Dict], filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

# Generate Data
train_data = [generate_example(i) for i in range(NUM_TRAIN)]
dev_data = [generate_example(i) for i in range(NUM_TRAIN, NUM_TRAIN + NUM_DEV)]

# Save
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_jsonl(train_data, f"{OUTPUT_DIR}/train.jsonl")
save_jsonl(dev_data, f"{OUTPUT_DIR}/dev.jsonl")

print(f"Generated {len(train_data)} training samples and {len(dev_data)} dev samples in '{OUTPUT_DIR}/'.")