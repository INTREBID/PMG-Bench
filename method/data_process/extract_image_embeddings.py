#!/usr/bin/env python3
"""
Extract CLIP embeddings for all images in FLICKR-AES dataset
"""

import os
import sys

import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import glob
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count, set_start_method
import argparse

# Set multiprocessing start method to spawn (avoid segfault from fork)
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass  # If already set, ignore error

# Configuration paths
DATASET_PATH = "{FLICKR_AES_BASE_PATH}"
MODEL_PATH = "{CLIP_VIT_H14_MODEL_PATH}"
OUTPUT_DIR = "{FLICKR_AES_BASE_PATH}/embeddings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "image_embeddings.npy")
METADATA_FILE = os.path.join(OUTPUT_DIR, "image_paths.json")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _get_processor():
    """Get processor in worker process (avoid pickle issues)"""
    # Use environment variable to pass model_path (global variables not available in spawn mode)
    model_path = os.environ.get('CLIP_MODEL_PATH', None)
    if model_path is None:
        return None
    try:
        from transformers import CLIPProcessor
        processor = CLIPProcessor.from_pretrained(model_path)
        return processor
    except Exception as e:
        print(f"Warning: worker process cannot create processor: {e}")
        return None

class ImageDataset(Dataset):
    """Image dataset class for DataLoader"""
    def __init__(self, image_paths, use_processor_in_dataset=False):
        self.image_paths = image_paths
        self.use_processor_in_dataset = use_processor_in_dataset
        # If processing in Dataset is needed, each worker will create its own processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            
            # If processing in Dataset is needed (multiprocessing mode)
            if self.use_processor_in_dataset:
                processor = _get_processor()
                if processor is not None:
                    try:
                        inputs = processor(images=img, return_tensors="pt")
                        # Remove batch dimension, DataLoader will add it automatically
                        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                        return inputs, img_path
                    except Exception as e:
                        print(f"Warning: Error processing image {img_path}: {e}")
                        return None, img_path
                else:
                    return None, img_path
            else:
                # Don't process, return raw image (single process mode, process in collate_fn)
                return img, img_path
        except Exception as e:
            # Return None to indicate failure, will be filtered later
            return None, img_path

def find_all_images(dataset_path):
    """Find all image files in dataset"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    
    for ext in image_extensions:
        pattern = os.path.join(dataset_path, '**', ext)
        image_paths.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and sort
    image_paths = sorted(list(set(image_paths)))
    return image_paths

def load_model(model_path):
    """Load CLIP model"""
    print(f"Loading model: {model_path}")
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    
    # Set to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded to device: {device}")
    
    return model, processor, device

def _collate_fn_processed(batch):
    """Collate function: Process preprocessed images (inputs format)"""
    valid_batch = [(inputs, path) for inputs, path in batch if inputs is not None]
    if not valid_batch:
        return None, []
    
    inputs_list, paths = zip(*valid_batch)
    # Merge inputs
    batch_inputs = {}
    for key in inputs_list[0].keys():
        batch_inputs[key] = torch.stack([inp[key] for inp in inputs_list])
    
    return batch_inputs, list(paths)

def _collate_fn_images(batch):
    """Collate function: Process raw images (PIL format)"""
    valid_batch = [(img, path) for img, path in batch if img is not None]
    if not valid_batch:
        return None, []
    
    images, paths = zip(*valid_batch)
    # Use processor to batch process images
    # Note: This function needs processor, but processor may not be pickleable
    # So if using multiprocessing, should process in Dataset
    return list(images), list(paths)

def extract_embeddings(model, processor, device, image_paths, batch_size=128, num_workers=8, model_path=None):
    """Extract image embeddings using DataLoader in parallel"""
    all_embeddings = []
    valid_paths = []
    
    print(f"Starting to process {len(image_paths)} images...")
    print(f"Using {num_workers} worker processes to load images in parallel")
    print(f"Batch size: {batch_size}")
    
    # Create dataset
    # If using multiprocessing, process images in Dataset (avoid processor pickle issues)
    # If using single process, can process in collate_fn (more efficient)
    if num_workers > 0:
        # Multiprocessing mode: process images in Dataset
        # Use environment variable to pass model_path (global variables not available in spawn mode)
        if model_path:
            os.environ['CLIP_MODEL_PATH'] = model_path
            pass
        else:
            pass
            num_workers = 0
        
        if num_workers > 0:
            dataset = ImageDataset(image_paths, use_processor_in_dataset=True)
            collate_fn = _collate_fn_processed
        else:
            # Fall back to single process mode
            dataset = ImageDataset(image_paths, use_processor_in_dataset=False)
            def collate_fn(batch):
                valid_batch = [(img, path) for img, path in batch if img is not None]
                if not valid_batch:
                    return None, []
                images, paths = zip(*valid_batch)
                try:
                    batch_inputs = processor(images=list(images), return_tensors="pt", padding=True)
                    return batch_inputs, list(paths)
                except Exception as e:
                    print(f"Warning: Error in collate processing: {e}")
                    return None, []
    else:
        # Single process mode: process images in collate_fn (more efficient)
        dataset = ImageDataset(image_paths, use_processor_in_dataset=False)
        # Create a collate function that uses processor
        def collate_fn(batch):
            valid_batch = [(img, path) for img, path in batch if img is not None]
            if not valid_batch:
                return None, []
            images, paths = zip(*valid_batch)
            try:
                batch_inputs = processor(images=list(images), return_tensors="pt", padding=True)
                return batch_inputs, list(paths)
            except Exception as e:
                print(f"Warning: Error in collate processing: {e}")
                return None, []
    
    # Use multiprocessing for acceleration (spawn mode is safer)
    use_persistent_workers = num_workers > 0
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True if device.type == 'cuda' else False,
        'persistent_workers': use_persistent_workers,
    }
    
    # prefetch_factor can only be used when num_workers > 0
    if num_workers > 0:
        dataloader_kwargs['multiprocessing_context'] = 'spawn'
        dataloader_kwargs['prefetch_factor'] = 2  # Prefetch more batches
    
    dataloader = DataLoader(**dataloader_kwargs)
    
    # Batch processing
    failed_count = 0
    for batch_inputs, batch_paths in tqdm(dataloader, desc="Extracting embeddings"):
        if batch_inputs is None:
            failed_count += len(batch_paths) if batch_paths else 0
            continue
        
        try:
            # Move to device
            inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            
            # Extract embeddings
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                # Normalize embeddings
                embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
            
            # Move to CPU and convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            all_embeddings.append(embeddings_np)
            valid_paths.extend(batch_paths)
            
        except Exception as e:
            print(f"Warning: Error processing batch: {e}")
            failed_count += len(batch_paths) if batch_paths else 0
            continue
    
    if failed_count > 0:
        print(f"Warning: {failed_count} images failed to process")
    
    # Merge all embeddings
    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)
        print(f"Successfully extracted embeddings for {len(valid_paths)} images")
        print(f"Embedding shape: {all_embeddings.shape}")
    else:
        print("Error: No embeddings successfully extracted")
        return None, []
    
    return all_embeddings, valid_paths

def main():
    parser = argparse.ArgumentParser(description='Extract CLIP embeddings for all images in FLICKR-AES dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH, help='Dataset path')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Model path')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Output directory')
    
    args = parser.parse_args()
    
    # Auto-detect worker count
    if args.num_workers is None:
        # Use more workers to speed up processing
        args.num_workers = min(cpu_count(), 8)  # Maximum 8 workers
    
    print("=" * 60)
    print("FLICKR-AES Image Embedding Extraction Tool")
    print("=" * 60)
    print(f"Configuration: batch_size={args.batch_size}, num_workers={args.num_workers}")
    
    # Find all images (including all subdirectories, such as processed_dataset, etc.)
    print("\n1. Finding all images in dataset...")
    print(f"Search path: {args.dataset_path}")
    print("Note: Will recursively search all subdirectories, including processed_dataset, etc.")
    image_paths = find_all_images(args.dataset_path)
    print(f"Found {len(image_paths)} images")
    
    # Display some statistics
    path_counts = {}
    for path in image_paths:
        # Get first-level subdirectory of path
        rel_path = os.path.relpath(path, args.dataset_path)
        first_dir = rel_path.split(os.sep)[0] if os.sep in rel_path else "root"
        path_counts[first_dir] = path_counts.get(first_dir, 0) + 1
    
    print("\nImage distribution statistics:")
    for dir_name, count in sorted(path_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dir_name}: {count} images")
    
    # Load model
    print("\n2. Loading CLIP model...")
    model, processor, device = load_model(args.model_path)
    
    # Extract embeddings, fall back to single process if multiprocessing fails
    print("\n3. Extracting image embeddings...")
    try:
        embeddings, valid_paths = extract_embeddings(
            model, processor, device, image_paths,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model_path=args.model_path
        )
    except Exception as e:
        if args.num_workers > 0:
            pass
            embeddings, valid_paths = extract_embeddings(
                model, processor, device, image_paths,
                batch_size=args.batch_size,
                num_workers=0
            )
        else:
            raise
    
    if embeddings is not None:
        # Update output paths
        output_file = os.path.join(args.output_dir, "image_embeddings.npy")
        metadata_file = os.path.join(args.output_dir, "image_paths.json")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save embeddings
        print(f"\n4. Saving embeddings to {output_file}...")
        np.save(output_file, embeddings)
        print(f"Embeddings saved: {embeddings.shape}")
        
        # Save image path metadata
        print(f"\n5. Saving image path metadata to {metadata_file}...")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(valid_paths, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(valid_paths)} image paths")
        
        # Save statistics
        stats = {
            "total_images_found": len(image_paths),
            "successful_extractions": len(valid_paths),
            "embedding_shape": list(embeddings.shape),
            "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else None,
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers
        }
        stats_file = os.path.join(args.output_dir, "extraction_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to {stats_file}")
        
        print("\n" + "=" * 60)
        print("Completed!")
        print("=" * 60)
    else:
        print("\nError: Extraction failed")

if __name__ == "__main__":
    main()

