#!/usr/bin/env python3

import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

# Configuration
LLM_PATH = "{LLM_MODEL_PATH}"
PROCESSED_DIR = "{SER_DATASET_BASE_PATH}/processed"
OUTPUT_DIR = "{SER_DATASET_BASE_PATH}/processed_masked"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
PROMPT_TEMPLATE = """Replace appearance features (like blonde, anime girl, character, color, style) with [V]. Keep actions (sits, stands, holds), expressions (sad expression, happy face), and environments (wavy-patterned surface, background).

Input: "{caption}"
Output:"""


def load_model():
    print(f"Loading model from {LLM_PATH}...")
    
    try:
        print("Trying AutoTokenizer with trust_remote_code...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
        print("✓ Loaded using AutoTokenizer")
    except Exception as e1:
        print(f"  Failed: {e1}")
        try:
            print("Trying to import Qwen2Tokenizer directly...")
            from transformers import Qwen2Tokenizer
            tokenizer = Qwen2Tokenizer.from_pretrained(LLM_PATH)
            print("✓ Loaded using Qwen2Tokenizer")
        except Exception as e2:
            print(f"  Failed: {e2}")
            raise ValueError(f"Failed to load tokenizer. Please update transformers: pip install --upgrade transformers")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
    
    print("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            LLM_PATH,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            trust_remote_code=True
        )
        print("✓ Model loaded successfully with trust_remote_code")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure transformers version >= 4.51.0 and trust_remote_code=True")
        raise
    
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully")
    return tokenizer, model


def mask_caption_with_llm(caption: str, tokenizer, model) -> str:
    if not caption or not caption.strip():
        return ""
    
    prompt = PROMPT_TEMPLATE.format(caption=caption)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Output:" in generated_text:
        output = generated_text.split("Output:")[-1].strip()
    elif "Output:" in generated_text:
        output = generated_text.split("Output:")[-1].strip()
    else:
        lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
        output = lines[-1] if lines else generated_text.strip()
    
    output = output.strip('"').strip("'").strip()
    
    if not output or len(output) > len(caption) * 2 or len(output) < 5:
        return caption
    
    return output


def process_sample(sample: dict, tokenizer, model) -> dict:
    
    if 'target_caption' in sample and sample['target_caption']:
        try:
            masked_target = mask_caption_with_llm(sample['target_caption'], tokenizer, model)
            sample['target_masked_caption'] = masked_target
        except Exception as e:
            print(f"Error processing target_caption: {e}")
            sample['target_masked_caption'] = sample['target_caption']
    else:
        sample['target_masked_caption'] = ""
    
    if 'history_items_info' in sample:
        for hist_item in sample['history_items_info']:
            if 'caption' in hist_item and hist_item['caption']:
                try:
                    masked_caption = mask_caption_with_llm(hist_item['caption'], tokenizer, model)
                    hist_item['masked_caption'] = masked_caption
                except Exception as e:
                    print(f"Error processing history caption: {e}")
                    hist_item['masked_caption'] = hist_item.get('caption', '')
            else:
                hist_item['masked_caption'] = ""
    
    return sample


def process_dataset(split: str, tokenizer, model):
    input_path = os.path.join(PROCESSED_DIR, f"{split}.json")
    output_path = os.path.join(OUTPUT_DIR, f"{split}.json")
    
    print(f"\n{'='*80}")
    print(f"Processing {split}.json...")
    print(f"{'='*80}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    processed_data = []
    for i, sample in enumerate(data):
        if (i + 1) % 10 == 0:
            print(f"  Processing sample {i+1}/{len(data)}...")
        
        processed_sample = process_sample(sample.copy(), tokenizer, model)
        processed_data.append(processed_sample)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_path}")
    
    print(f"\nSample results (first 2):")
    for i, sample in enumerate(processed_data[:2]):
        print(f"\n  [{i+1}] Target:")
        print(f"      Original: {sample.get('target_caption', '')[:100]}...")
        print(f"      Masked:   {sample.get('target_masked_caption', '')[:100]}...")
        if sample.get('history_items_info') and len(sample['history_items_info']) > 0:
            hist = sample['history_items_info'][0]
            print(f"      History (first):")
            print(f"        Original: {hist.get('caption', '')[:100]}...")
            print(f"        Masked:   {hist.get('masked_caption', '')[:100]}...")


def main():
    print("=" * 80)
    print("Masking Captions with LLM")
    print("=" * 80)
    
    tokenizer, model = load_model()
    
    for split in ['train', 'val', 'test']:
        input_path = os.path.join(PROCESSED_DIR, f"{split}.json")
        if os.path.exists(input_path):
            process_dataset(split, tokenizer, model)
        else:
            print(f"Warning: {input_path} not found, skipping...")
    
    print("\n" + "=" * 80)
    print("Processing completed!")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

