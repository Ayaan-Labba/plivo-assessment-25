import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os

# --- CONFIGURATION ---
# Strict threshold for PII to ensure High Precision (minimizes False Positives)
PII_THRESHOLD = 0.0
# Normal threshold for non-sensitive entities (optional, can stay 0.0)
DEFAULT_THRESHOLD = 0.0

def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:
            continue
            
        label = ID2LABEL.get(int(lid), "O")
        
        # BIO Logic
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                # Broken sequence or new entity
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Load Tokenizer & Model
    load_path = args.model_dir if args.model_name is None else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    # tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=True)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}
    
    # Pre-calculate the ID for "O" (Outside) to use as a fallback
    # Assuming standard BIO scheme where 'O' exists.
    O_LABEL_ID = -1
    for _id, _lab in ID2LABEL.items():
        if _lab == "O":
            O_LABEL_ID = _id
            break
    
    if O_LABEL_ID == -1:
        # Fallback if "O" isn't explicitly found (rare)
        O_LABEL_ID = 0 

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]  # Shape: (seq_len, num_labels)
                
                # --- NEW LOGIC START ---
                
                # 1. Get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # 2. Get top prediction and its confidence
                confidences, pred_ids = torch.max(probs, dim=-1)
                
                # 3. Apply Thresholding
                final_preds = []
                for idx, pred_id_tensor in enumerate(pred_ids):
                    pred_id = pred_id_tensor.item()
                    confidence = confidences[idx].item()
                    label_str = ID2LABEL.get(pred_id, "O")
                    
                    # Check if this is a PII label we need to be careful about
                    # We strip B- or I- prefix to check the category
                    category = label_str.split("-")[1] if "-" in label_str else label_str
                    
                    # Check if it is PII (using your helper or manual list)
                    # Note: You need to ensure label_is_pii handles "B-EMAIL" vs "EMAIL" correctly
                    # or just pass the full label if your helper expects that.
                    is_pii_category = label_is_pii(category) or label_is_pii(label_str)

                    if is_pii_category:
                        if confidence < PII_THRESHOLD:
                            final_preds.append(O_LABEL_ID) # Drop low confidence PII
                        else:
                            final_preds.append(pred_id)
                    else:
                        # Non-PII entities (CITY, LOCATION) or 'O'
                        if confidence < DEFAULT_THRESHOLD:
                            final_preds.append(O_LABEL_ID)
                        else:
                            final_preds.append(pred_id)
                            
                # --- NEW LOGIC END ---

            spans = bio_to_spans(text, offsets, final_preds)
            
            ents = []
            for s, e, lab in spans:
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")

if __name__ == "__main__":
    main()