#!/usr/bin/env python3
import os
import sys
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

LLM_PATH = "{LLM_MODEL_PATH}"
POG_BASE_DIR = "{POG_BASE_PATH}"
INPUT_CAPTIONS_FILE = os.path.join(POG_BASE_DIR, "POG_captions_sampled.json")
OUTPUT_CAPTIONS_FILE = os.path.join(POG_BASE_DIR, "POG_captions_sampled_masked.json")

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    print(f"Using device: {DEVICE}")
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
    print(f"Batch size: {BATCH_SIZE}")
    
    if gpu_memory >= 40:
        if BATCH_SIZE < 64:
            BATCH_SIZE = 64
            print(f"  -> Adjusted batch size to {BATCH_SIZE} for large GPU memory")
else:
    print(f"Using device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")

PROMPT_TEMPLATE = """Replace only appearance features (color, style, material, pattern, brand, design details) with [V]. Keep the item type (hoodie, dress, jeans, etc.) and sentence structure intact. Only output the masked sentence, nothing else.

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


def mask_captions_batch(captions: list, tokenizer, model, debug=False, batch_idx=0) -> list:
    if not captions:
        return []
    
    results = [""] * len(captions)
    
    valid_indices = []
    valid_captions = []
    prompts = []
    
    for i, caption in enumerate(captions):
        if not caption or not caption.strip():
            results[i] = ""
            continue
        valid_indices.append(i)
        valid_captions.append(caption)
        prompts.append(PROMPT_TEMPLATE.format(caption=caption))
    
    if not prompts:
        return results
    
    
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
        )
    
    stats = {
        'total': len(valid_indices),
        'has_output_marker': 0,
        'no_output_marker': 0,
        'has_v_mask': 0,
        'no_v_mask': 0,
        'returned_original': 0,
        'success': 0
    }
    
    for idx, (original_idx, original_caption, output_ids) in enumerate(zip(valid_indices, valid_captions, outputs)):
        full_generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        
        if "Output:" in full_generated_text:
            output_section = full_generated_text.split("Output:")[-1]
            
            if "Input:" in output_section:
                output = output_section.split("Input:")[0].strip()
            else:
                output = output_section.strip()
            stats['has_output_marker'] += 1
        elif "Output:" in full_generated_text:
            output_section = full_generated_text.split("Output:")[-1]
            if "Input:" in output_section:
                output = output_section.split("Input:")[0].strip()
            else:
                output = output_section.strip()
            stats['has_output_marker'] += 1
        else:
            input_length = inputs['input_ids'][idx].shape[0]
            generated_ids = output_ids[input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if "Input:" in generated_text:
                output = generated_text.split("Input:")[0].strip()
            else:
                lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
                output = lines[-1] if lines else generated_text.strip()
            stats['no_output_marker'] += 1
            
        
        original_output = output
        output = output.strip('"').strip("'").strip()
        
        output = output.replace('\n', ' ').replace('\r', ' ')
        output = ' '.join(output.split())
        
        if "Input:" in output:
            output = output.split("Input:")[0].strip()
        if "Input:" in output:
            output = output.split("Input:")[0].strip()
        if "Output:" in output and output.count("Output:") > 1:
            # If there are multiple Output:, take the last one
            output = output.split("Output:")[-1].strip()
        
        output = output.rstrip('[').rstrip(' ').strip()
        
        has_v_mask = "[V]" in output
        
        output_stripped = output.strip()
        
        is_invalid = False
        
        
        if not output_stripped or len(output_stripped) <= 2:
            is_invalid = True
        elif output_stripped in ["V", "[V]", "v", "v.", "V.", "[V]."]:
            is_invalid = True
        elif output_stripped.replace("V", "").replace("[", "").replace("]", "").replace(" ", "").strip() == "":
            is_invalid = True
        elif "\n" in output or "Input:" in output:
            is_invalid = True
        elif output_stripped.endswith("[") or output_stripped.endswith("\n"):
            is_invalid = True
        
        if is_invalid:
            output = original_caption
            stats['returned_original'] += 1
        elif len(output) > len(original_caption) * 2:
            
            
            for end_char in ['.', '!', '?']:
                if end_char in output:
                    output = output.split(end_char)[0] + end_char
                    break
            if len(output) > len(original_caption) * 2:
                output = original_caption
                stats['returned_original'] += 1
        else:
            if has_v_mask:
                stats['has_v_mask'] += 1
                stats['success'] += 1
            else:
                stats['no_v_mask'] += 1
        
        results[original_idx] = output
    
    
    return results


def process_captions(input_file: str, output_file: str, tokenizer, model):
    print(f"\n{'='*80}")
    print(f"Processing captions from {input_file}...")
    print(f"{'='*80}")
    
    print("Loading captions...")
    with open(input_file, 'r', encoding='utf-8') as f:
        captions_dict = json.load(f)
    
    total_items = len(captions_dict)
    print(f"Total items: {total_items}")
    
    masked_captions_dict = {}
    processed_count = 0
    error_count = 0
    
    item_ids = list(captions_dict.keys())
    captions_list = [captions_dict[item_id] for item_id in item_ids]
    
    global_stats = {
        'total_processed': 0,
        'has_v_mask': 0,
        'no_v_mask': 0,
        'returned_original': 0,
        'empty_output': 0
    }
    
    num_batches = (len(captions_list) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {len(captions_list)} captions in {num_batches} batches (batch_size={BATCH_SIZE})...")
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(captions_list))
        
        batch_item_ids = item_ids[start_idx:end_idx]
        batch_captions = captions_list[start_idx:end_idx]
        
        try:
            debug_mode = (batch_idx == 0)
            batch_results = mask_captions_batch(batch_captions, tokenizer, model, debug=debug_mode, batch_idx=batch_idx)
            
            for item_id, masked_caption in zip(batch_item_ids, batch_results):
                masked_captions_dict[item_id] = masked_caption
                processed_count += 1
                global_stats['total_processed'] += 1
                
                if not masked_caption:
                    global_stats['empty_output'] += 1
                elif "[V]" in masked_caption:
                    global_stats['has_v_mask'] += 1
                elif masked_caption in captions_dict.get(item_id, ""):
                    global_stats['returned_original'] += 1
                else:
                    global_stats['no_v_mask'] += 1
                
        except Exception as e:
            print(f"\n[ERROR] Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            for item_id, caption in zip(batch_item_ids, batch_captions):
                masked_captions_dict[item_id] = caption if caption else ""
                error_count += 1
    
    print(f"\nSaving masked captions to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(masked_captions_dict, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved to {output_file}")
    print(f"\nStatistics:")
    print(f"  - Total items: {total_items}")
    print(f"  - Successfully processed: {processed_count}")
    print(f"  - Errors: {error_count}")
    print(f"\nMasking Statistics:")
    print(f"  - Items with [V] mask: {global_stats['has_v_mask']} ({global_stats['has_v_mask']/processed_count*100:.1f}%)")
    print(f"  - Items without [V] mask: {global_stats['no_v_mask']} ({global_stats['no_v_mask']/processed_count*100:.1f}%)")
    print(f"  - Items returned original: {global_stats['returned_original']} ({global_stats['returned_original']/processed_count*100:.1f}%)")
    print(f"  - Empty outputs: {global_stats['empty_output']} ({global_stats['empty_output']/processed_count*100:.1f}%)")
    
    print(f"\nSample results (first 5 items):")
    sample_items = list(captions_dict.items())[:5]
    for item_id, original_caption in sample_items:
        masked_caption = masked_captions_dict.get(item_id, "")
        print(f"\n  Item ID: {item_id}")
        print(f"    Original: {original_caption[:100]}...")
        print(f"    Masked:   {masked_caption[:100]}...")


def main():
    print("=" * 80)
    print("Masking POG Captions with LLM")
    print("=" * 80)
    
    if not os.path.exists(INPUT_CAPTIONS_FILE):
        print(f"Error: Input file not found: {INPUT_CAPTIONS_FILE}")
        sys.exit(1)
    
    tokenizer, model = load_model()
    process_captions(INPUT_CAPTIONS_FILE, OUTPUT_CAPTIONS_FILE, tokenizer, model)
    
    print("\n" + "=" * 80)
    print("Processing completed!")
    print("=" * 80)
    print(f"Output file: {OUTPUT_CAPTIONS_FILE}")


if __name__ == "__main__":
    main()

