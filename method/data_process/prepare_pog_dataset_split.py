#!/usr/bin/env python3

import os
import json
import random
from typing import List, Dict, Tuple
from collections import defaultdict

RANDOM_SEED = 42
POG_BASE_DIR = "{POG_BASE_PATH}"
USER_DATA_FILE = os.path.join(POG_BASE_DIR, "POG_subset_user_2000.txt")
CAPTIONS_FILE = os.path.join(POG_BASE_DIR, "POG_captions_sampled.json")
USER_STYLES_FILE = os.path.join(POG_BASE_DIR, "POG_user_styles.json")
IMAGES_DIR = os.path.join(POG_BASE_DIR, "images_sampled")
OUTPUT_DIR = os.path.join(POG_BASE_DIR, "processed_dataset")
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "train.json")
OUTPUT_VAL = os.path.join(OUTPUT_DIR, "val.json")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "test.json")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.05
TEST_RATIO = 0.15


def load_user_data() -> Dict[str, Dict]:
    print("=" * 80)
    print("Loading user interaction data...")
    print("=" * 80)
    
    if not os.path.exists(USER_DATA_FILE):
        raise FileNotFoundError(f"User data file not found at {USER_DATA_FILE}")
    
    user_data = {}
    
    print(f"  - Reading file: {USER_DATA_FILE}")
    with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split(',')
                if len(parts) < 2:
                    print(f"  - Warning: Line {line_num} has invalid format, skipping...")
                    continue
                
                user_id = parts[0].strip()
                item_ids_str = parts[1].strip()
                
                item_ids = [item_id.strip() for item_id in item_ids_str.split(';') if item_id.strip()]
                
                if not item_ids:
                    continue
                
                target_item_id = None
                history_item_ids = item_ids.copy()
                
                if len(item_ids) >= 2:
                    target_item_id = item_ids[-2]
                    history_item_ids = item_ids[:-1]
                elif len(item_ids) == 1:
                    history_item_ids = item_ids
                    target_item_id = None
                
                user_data[user_id] = {
                    'history_item_ids': history_item_ids,
                    'target_item_id': target_item_id
                }
                
            except Exception as e:
                print(f"  - Warning: Error parsing line {line_num}: {e}, skipping...")
                continue
    
    print(f"  - Loaded {len(user_data):,} users")
    total_interactions = sum(len(data['history_item_ids']) for data in user_data.values())
    avg_interactions = total_interactions / len(user_data) if user_data else 0
    max_interactions = max(len(data['history_item_ids']) for data in user_data.values()) if user_data else 0
    min_interactions = min(len(data['history_item_ids']) for data in user_data.values()) if user_data else 0
    
    users_with_target = sum(1 for data in user_data.values() if data.get('target_item_id') is not None)
    
    print(f"  - Total interactions (after removing last item): {total_interactions:,}")
    print(f"  - Avg interactions per user: {avg_interactions:.1f}")
    print(f"  - Max interactions per user: {max_interactions}")
    print(f"  - Min interactions per user: {min_interactions}")
    print(f"  - Users with target item: {users_with_target:,} ({users_with_target/len(user_data)*100:.1f}%)")
    
    return user_data


def load_captions() -> Dict[str, str]:
    print("\n" + "=" * 80)
    print("Loading item captions...")
    print("=" * 80)
    
    if not os.path.exists(CAPTIONS_FILE):
        print(f"  - Warning: Captions file not found at {CAPTIONS_FILE}, returning empty dict")
        return {}
    
    with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    
    print(f"  - Loaded {len(captions):,} captions")
    return captions


def load_user_styles() -> Dict[str, str]:
    print("\n" + "=" * 80)
    print("Loading user styles...")
    print("=" * 80)
    
    if not os.path.exists(USER_STYLES_FILE):
        print(f"  - Warning: User styles file not found at {USER_STYLES_FILE}, returning empty dict")
        return {}
    
    with open(USER_STYLES_FILE, 'r', encoding='utf-8') as f:
        styles_data = json.load(f)
    
    user_styles = {}
    if isinstance(styles_data, list):
        for item in styles_data:
            if 'user' in item and 'style' in item:
                user_styles[item['user']] = item['style']
    elif isinstance(styles_data, dict):
        user_styles = styles_data
    
    print(f"  - Loaded styles for {len(user_styles):,} users")
    return user_styles


def find_image_path(item_id: str) -> str:
    if not os.path.exists(IMAGES_DIR):
        return None
    
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_path = os.path.join(IMAGES_DIR, item_id + ext)
        if os.path.exists(image_path):
            return image_path
    
    return None


def create_dataset_samples(user_data: Dict[str, Dict], captions: Dict[str, str], 
                          user_styles: Dict[str, str]) -> List[Dict]:
    print("\n" + "=" * 80)
    print("Creating dataset samples...")
    print("=" * 80)
    
    samples = []
    
    for user_id, data in user_data.items():
        history_item_ids = data['history_item_ids']
        target_item_id = data.get('target_item_id')
        
        history_items_info = []
        for item_id in history_item_ids:
            caption = captions.get(item_id, "")
            image_path = find_image_path(item_id)
            
            history_items_info.append({
                'item_id': item_id,
                'caption': caption,
                'image_path': image_path
            })
        
        target_item_info = None
        if target_item_id:
            target_caption = captions.get(target_item_id, "")
            target_image_path = find_image_path(target_item_id)
            target_item_info = {
                'item_id': target_item_id,
                'caption': target_caption,
                'image_path': target_image_path
            }
        
        user_style = user_styles.get(user_id, "")
        
        sample = {
            'user_id': user_id,
            'history_item_ids': history_item_ids,
            'history_items_info': history_items_info,
            'target_item_id': target_item_id,
            'target_item_info': target_item_info,
            'user_style': user_style,
            'num_interactions': len(history_item_ids)
        }
        
        samples.append(sample)
    
    print(f"  - Created {len(samples)} samples")
    avg_interactions = sum(s['num_interactions'] for s in samples) / len(samples) if samples else 0
    print(f"  - Average interactions per user: {avg_interactions:.1f}")
    
    samples_with_images = sum(1 for s in samples 
                              if any(item.get('image_path') for item in s['history_items_info']))
    print(f"  - Samples with images: {samples_with_images}/{len(samples)} ({samples_with_images/len(samples)*100:.1f}%)")
    
    return samples


def split_dataset(samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    print("\n" + "=" * 80)
    print("Splitting dataset by users...")
    print("=" * 80)
    
    random.seed(RANDOM_SEED)
    
    user_samples = {sample['user_id']: sample for sample in samples}
    user_ids = list(user_samples.keys())
    random.shuffle(user_ids)
    
    total_users = len(user_ids)
    train_size = int(total_users * TRAIN_RATIO)
    val_size = int(total_users * VAL_RATIO)
    
    train_user_ids = set(user_ids[:train_size])
    val_user_ids = set(user_ids[train_size:train_size + val_size])
    test_user_ids = set(user_ids[train_size + val_size:])
    
    train_samples = [user_samples[uid] for uid in train_user_ids]
    val_samples = [user_samples[uid] for uid in val_user_ids]
    test_samples = [user_samples[uid] for uid in test_user_ids]
    
    print(f"  - Total users: {total_users}")
    print(f"  - Train users: {len(train_user_ids)} ({len(train_user_ids)/total_users*100:.1f}%)")
    print(f"  - Val users: {len(val_user_ids)} ({len(val_user_ids)/total_users*100:.1f}%)")
    print(f"  - Test users: {len(test_user_ids)} ({len(test_user_ids)/total_users*100:.1f}%)")
    
    train_interactions = sum(s['num_interactions'] for s in train_samples)
    val_interactions = sum(s['num_interactions'] for s in val_samples)
    test_interactions = sum(s['num_interactions'] for s in test_samples)
    total_interactions = train_interactions + val_interactions + test_interactions
    
    print(f"\n  - Total interactions: {total_interactions:,}")
    print(f"  - Train interactions: {train_interactions:,} ({train_interactions/total_interactions*100:.1f}%)")
    print(f"  - Val interactions: {val_interactions:,} ({val_interactions/total_interactions*100:.1f}%)")
    print(f"  - Test interactions: {test_interactions:,} ({test_interactions/total_interactions*100:.1f}%)")
    
    return train_samples, val_samples, test_samples


def save_datasets(train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
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


def print_statistics(train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    all_samples = train_samples + val_samples + test_samples
    
    print(f"\n[1] Sample Counts:")
    print(f"  - Train: {len(train_samples):,}")
    print(f"  - Val: {len(val_samples):,}")
    print(f"  - Test: {len(test_samples):,}")
    print(f"  - Total: {len(all_samples):,}")
    
    print(f"\n[2] Interaction Statistics:")
    train_interactions = [s['num_interactions'] for s in train_samples]
    val_interactions = [s['num_interactions'] for s in val_samples]
    test_interactions = [s['num_interactions'] for s in test_samples]
    all_interactions = train_interactions + val_interactions + test_interactions
    
    print(f"  - Train - Avg: {sum(train_interactions)/len(train_interactions):.1f}, "
          f"Max: {max(train_interactions)}, Min: {min(train_interactions)}")
    print(f"  - Val - Avg: {sum(val_interactions)/len(val_interactions):.1f}, "
          f"Max: {max(val_interactions)}, Min: {min(val_interactions)}")
    print(f"  - Test - Avg: {sum(test_interactions)/len(test_interactions):.1f}, "
          f"Max: {max(test_interactions)}, Min: {min(test_interactions)}")
    print(f"  - Overall - Avg: {sum(all_interactions)/len(all_interactions):.1f}, "
          f"Max: {max(all_interactions)}, Min: {min(all_interactions)}")
    
    print(f"\n[3] Sample Data Structure:")
    if all_samples:
        sample = all_samples[0]
        print(f"  - Keys: {list(sample.keys())}")
        print(f"  - user_id: {sample['user_id']}")
        print(f"  - num_interactions: {sample['num_interactions']}")
        print(f"  - has_target_item: {sample.get('target_item_id') is not None}")
        print(f"  - has_user_style: {bool(sample.get('user_style'))}")
        print(f"  - First 3 history items:")
        for i, item in enumerate(sample['history_items_info'][:3]):
            print(f"      [{i+1}] item_id: {item['item_id']}, "
                  f"has_caption: {bool(item.get('caption'))}, "
                  f"has_image: {bool(item.get('image_path'))}")


def main():
    print("=" * 80)
    print("POG Dataset Processing and Splitting")
    print("=" * 80)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Split ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    
    user_data = load_user_data()
    captions = load_captions()
    user_styles = load_user_styles()
    samples = create_dataset_samples(user_data, captions, user_styles)
    train_samples, val_samples, test_samples = split_dataset(samples)
    save_datasets(train_samples, val_samples, test_samples)
    print_statistics(train_samples, val_samples, test_samples)
    
    print("\n" + "=" * 80)
    print("Dataset processing completed!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_TRAIN}")
    print(f"  - {OUTPUT_VAL}")
    print(f"  - {OUTPUT_TEST}")


if __name__ == "__main__":
    main()

