#!/usr/bin/env python3
"""
Process and split SER dataset for style transfer task

Dataset structure:
- History items: Each topic folder serves as a history image collection
- Target item: Randomly sample one image from other topic folders as target item
- Fixed random seed for reproducibility
"""

import os
import json
import random
from typing import List, Dict, Tuple
from collections import defaultdict

RANDOM_SEED = 42

BASE_PATH = "{SER_DATASET_BASE_PATH}"
IMAGES_ROOT = os.path.join(BASE_PATH, "Images")
ANNOTATIONS_PATH = os.path.join(BASE_PATH, "Annotations", "all_annos.json")
CAPTIONS_PATH = os.path.join(BASE_PATH, "Annotations", "ser30k_captions.json")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.05
TEST_RATIO = 0.15

OUTPUT_DIR = os.path.join(BASE_PATH, "processed")
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "train.json")
OUTPUT_VAL = os.path.join(OUTPUT_DIR, "val.json")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "test.json")


def load_annotations_and_captions():
    """Load annotations and caption data"""
    print("=" * 80)
    print("Loading annotations and captions...")
    print("=" * 80)
    
    with open(ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
        all_annos = json.load(f)
    
    annotations = all_annos.get('annotations', [])
    categories = all_annos.get('categories', {})
    
    print(f"  - Loaded {len(annotations)} annotations")
    print(f"  - Categories: {categories}")
    
    anno_dict = {}
    for anno in annotations:
        key = f"{anno['topic']}/{anno['file_name']}"
        anno_dict[key] = anno
    
    captions = {}
    if os.path.exists(CAPTIONS_PATH):
        with open(CAPTIONS_PATH, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        print(f"  - Loaded {len(captions)} English captions")
    else:
        print(f"  - Warning: {CAPTIONS_PATH} not found")
    
    return anno_dict, captions, categories


def get_topic_to_images_mapping():
    """Get all image files under each topic folder"""
    print("\n" + "=" * 80)
    print("Scanning Images directory...")
    print("=" * 80)
    
    topic_to_images = defaultdict(list)
    
    if not os.path.exists(IMAGES_ROOT):
        raise FileNotFoundError(f"Images directory not found: {IMAGES_ROOT}")
    
    topics = [d for d in os.listdir(IMAGES_ROOT) 
              if os.path.isdir(os.path.join(IMAGES_ROOT, d))]
    
    print(f"  - Found {len(topics)} topic folders")
    
    for topic in topics:
        topic_path = os.path.join(IMAGES_ROOT, topic)
        image_files = [f for f in os.listdir(topic_path) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg', '.gif'))]
        
        if len(image_files) > 0:
            topic_to_images[topic] = sorted(image_files)
    
    print(f"  - Topics with images: {len(topic_to_images)}")
    
    total_images = sum(len(images) for images in topic_to_images.values())
    print(f"  - Total images: {total_images}")
    
    image_counts = [len(images) for images in topic_to_images.values()]
    if image_counts:
        print(f"  - Min images per topic: {min(image_counts)}")
        print(f"  - Max images per topic: {max(image_counts)}")
        print(f"  - Avg images per topic: {sum(image_counts) / len(image_counts):.1f}")
    
    return topic_to_images


def create_dataset_samples(topic_to_images: Dict[str, List[str]], 
                           anno_dict: Dict[str, Dict],
                           captions: Dict[str, str]) -> List[Dict]:
    """
    Create dataset samples
    
    For each topic (as history items):
    - History item id: topic name
    - Randomly sample one image from other topics as target item
    """
    print("\n" + "=" * 80)
    print("Creating dataset samples...")
    print("=" * 80)
    
    random.seed(RANDOM_SEED)
    
    all_topics = sorted(list(topic_to_images.keys()))
    print(f"  - Total topics: {len(all_topics)}")
    
    samples = []
    skipped = 0
    
    for history_topic in all_topics:
        history_images = topic_to_images[history_topic]
        
        if len(history_images) == 0:
            skipped += 1
            continue
        
        history_item_ids = [f"{history_topic}/{img}" for img in history_images]
        
        other_topics = [t for t in all_topics if t != history_topic]
        
        if len(other_topics) == 0:
            skipped += 1
            continue
        
        target_topic = random.choice(other_topics)
        target_images = topic_to_images[target_topic]
        
        if len(target_images) == 0:
            skipped += 1
            continue
        
        target_file_name = random.choice(target_images)
        target_item_id = f"{target_topic}/{target_file_name}"
        
        target_anno = anno_dict.get(target_item_id, {})
        target_caption = captions.get(target_item_id, "")
        
        history_items_info = []
        for hist_id in history_item_ids:
            hist_anno = anno_dict.get(hist_id, {})
            hist_caption = captions.get(hist_id, "")
            history_items_info.append({
                "item_id": hist_id,
                "annotation": hist_anno,
                "caption": hist_caption,
                "image_path": os.path.join(IMAGES_ROOT, hist_id)
            })
        
        sample = {
            "history_topic": history_topic,
            "history_item_ids": history_item_ids,
            "history_items_info": history_items_info,
            "target_item_id": target_item_id,
            "target_annotation": target_anno,
            "target_caption": target_caption,
            "target_image_path": os.path.join(IMAGES_ROOT, target_item_id)
        }
        
        samples.append(sample)
    
    print(f"  - Created {len(samples)} samples")
    print(f"  - Skipped {skipped} topics (no images or no other topics)")
    
    return samples


def split_dataset(samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/val/test"""
    print("\n" + "=" * 80)
    print("Splitting dataset...")
    print("=" * 80)
    
    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    
    total = len(samples)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)
    
    train_samples = samples[:train_size]
    val_samples = samples[train_size:train_size + val_size]
    test_samples = samples[train_size + val_size:]
    
    print(f"  - Total samples: {total}")
    print(f"  - Train: {len(train_samples)} ({len(train_samples)/total*100:.1f}%)")
    print(f"  - Val: {len(val_samples)} ({len(val_samples)/total*100:.1f}%)")
    print(f"  - Test: {len(test_samples)} ({len(test_samples)/total*100:.1f}%)")
    
    return train_samples, val_samples, test_samples


def save_datasets(train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
    """Save datasets to JSON files"""
    print("\n" + "=" * 80)
    print("Saving datasets...")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    print(f"  - Saved train set: {OUTPUT_TRAIN} ({len(train_samples)} samples)")
    
    with open(OUTPUT_VAL, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    print(f"  - Saved val set: {OUTPUT_VAL} ({len(val_samples)} samples)")
    
    with open(OUTPUT_TEST, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)
    print(f"  - Saved test set: {OUTPUT_TEST} ({len(test_samples)} samples)")


def print_sample_statistics(train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
    """Print sample statistics"""
    print("\n" + "=" * 80)
    print("Sample Statistics")
    print("=" * 80)
    
    all_samples = train_samples + val_samples + test_samples
    
    total_history_items = sum(len(s['history_item_ids']) for s in all_samples)
    avg_history_items = total_history_items / len(all_samples) if all_samples else 0
    
    print(f"\nHistory Items:")
    print(f"  - Total history items across all samples: {total_history_items:,}")
    print(f"  - Average history items per sample: {avg_history_items:.1f}")
    
    print(f"\nSample Examples (first 2 from train set):")
    for i, sample in enumerate(train_samples[:2]):
        print(f"\n  [{i+1}] Sample:")
        print(f"      History topic: {sample['history_topic']}")
        print(f"      History items count: {len(sample['history_item_ids'])}")
        print(f"      Target item: {sample['target_item_id']}")
        print(f"      Target caption: {sample['target_caption'][:100]}..." if len(sample['target_caption']) > 100 else f"      Target caption: {sample['target_caption']}")


def main():
    """Main function"""
    print("=" * 80)
    print("SER Dataset Processing for Style Transfer")
    print("=" * 80)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Split ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    
    anno_dict, captions, categories = load_annotations_and_captions()
    
    topic_to_images = get_topic_to_images_mapping()
    
    samples = create_dataset_samples(topic_to_images, anno_dict, captions)
    
    train_samples, val_samples, test_samples = split_dataset(samples)
    
    save_datasets(train_samples, val_samples, test_samples)
    
    print_sample_statistics(train_samples, val_samples, test_samples)
    
    print("\n" + "=" * 80)
    print("Dataset processing completed!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_TRAIN}")
    print(f"  - {OUTPUT_VAL}")
    print(f"  - {OUTPUT_TEST}")


if __name__ == "__main__":
    main()

