#!/usr/bin/env python3
"""
Process FLICKR-AES dataset:
1. Read rating data for each user (worker)
2. Create item interaction sequence fields and scores for each user, including image paths
3. Split dataset by interactions: train_ratio: 0.7, val_ratio: 0.15, test_ratio: 0.15
   This ensures that training and test sets contain the same user groups
"""

import os
import json
import random
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict

# Fixed random seed
RANDOM_SEED = 42

# Dataset paths
FLICKR_AES_BASE_PATH = "{FLICKR_AES_BASE_PATH}"
CSV_FILE = os.path.join(FLICKR_AES_BASE_PATH, "iccv_17_aesthetics_dataset", " FLICKR-AES_image_labeled_by_each_worker.csv")
IMAGES_DIR = os.path.join(FLICKR_AES_BASE_PATH, "40K")

# Output paths
OUTPUT_DIR = os.path.join(FLICKR_AES_BASE_PATH, "processed_dataset")
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "train.json")
OUTPUT_VAL = os.path.join(OUTPUT_DIR, "val.json")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "test.json")

# Split ratios (by interactions)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_worker_data() -> pd.DataFrame:
    """Load user rating data"""
    print("=" * 80)
    print("Loading worker annotation data...")
    print("=" * 80)
    
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV file not found at {CSV_FILE}")
    
    # Read CSV file
    print(f"  - Reading CSV file: {CSV_FILE}")
    try:
        df = pd.read_csv(CSV_FILE, encoding='utf-8', on_bad_lines='skip', engine='python')
    except TypeError:
        df = pd.read_csv(CSV_FILE, encoding='utf-8', error_bad_lines=False, warn_bad_lines=True, engine='python')
    
    # Clean column names (remove spaces)
    df.columns = df.columns.str.strip()
    
    print(f"  - Loaded {len(df):,} annotation records")
    print(f"  - Columns: {list(df.columns)}")
    
    # Check required columns
    required_cols = ['worker', 'imagePair', 'score']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Available columns: {list(df.columns)}")
    
    # Clean data
    df = df.dropna(subset=['worker', 'imagePair', 'score'])
    df['worker'] = df['worker'].astype(str)
    df['imagePair'] = df['imagePair'].astype(str).str.strip()
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df.dropna(subset=['score'])
    
    print(f"  - After cleaning: {len(df):,} records")
    print(f"  - Unique workers: {df['worker'].nunique():,}")
    print(f"  - Unique images: {df['imagePair'].nunique():,}")
    print(f"  - Score range: [{df['score'].min()}, {df['score'].max()}]")
    print(f"  - Score distribution:")
    print(df['score'].value_counts().sort_index())
    
    return df


def create_user_interaction_sequences(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Create interaction sequences for each user"""
    print("\n" + "=" * 80)
    print("Creating user interaction sequences...")
    print("=" * 80)
    
    # Group by user
    user_interactions = defaultdict(list)
    
    # Group by user, preserve original order (or sort by some order)
    for _, row in df.iterrows():
        worker = str(row['worker'])
        image_pair = str(row['imagePair']).strip()
        score = float(row['score'])
        
        user_interactions[worker].append({
            'item_id': image_pair,
            'score': score
        })
    
    print(f"  - Total users: {len(user_interactions):,}")
    
    # Statistics
    interaction_counts = [len(interactions) for interactions in user_interactions.values()]
    print(f"  - Total interactions: {sum(interaction_counts):,}")
    print(f"  - Avg interactions per user: {sum(interaction_counts) / len(interaction_counts):.1f}")
    print(f"  - Max interactions per user: {max(interaction_counts)}")
    print(f"  - Min interactions per user: {min(interaction_counts)}")
    
    # Filter out users with too few interactions (need at least 2 interactions to form a sequence)
    filtered_interactions = {
        worker: interactions 
        for worker, interactions in user_interactions.items() 
        if len(interactions) >= 2
    }
    
    print(f"\n  - Users with >= 2 interactions: {len(filtered_interactions):,}")
    print(f"  - Filtered out {len(user_interactions) - len(filtered_interactions)} users with < 2 interactions")
    
    return filtered_interactions


def find_image_path(item_id: str) -> str:
    """Find image path"""
    image_path = None
    
    # Try to find image file
    if os.path.exists(IMAGES_DIR):
        # First check if item_id already contains extension
        item_id_lower = item_id.lower()
        has_ext = any(item_id_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])
        
        if has_ext:
            # If already contains extension, use directly
            full_path = os.path.join(IMAGES_DIR, item_id)
            if os.path.exists(full_path):
                image_path = full_path
        else:
            # If no extension, try adding various extensions
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                full_path = os.path.join(IMAGES_DIR, item_id + ext)
                if os.path.exists(full_path):
                    image_path = full_path
                    break
    
    return image_path


def create_dataset_samples(user_interactions: Dict[str, List[Dict]]) -> List[Dict]:
    """Create dataset samples with image paths"""
    print("\n" + "=" * 80)
    print("Creating dataset samples with image paths...")
    print("=" * 80)
    
    samples = []
    found_images = 0
    missing_images = 0
    
    for user_id, interactions in user_interactions.items():
        # Build interaction sequence (contains item ID, score, and image path)
        interaction_sequence = []
        for interaction in interactions:
            item_id = interaction['item_id']
            score = interaction['score']
            image_path = find_image_path(item_id)
            
            if image_path:
                found_images += 1
            else:
                missing_images += 1
            
            interaction_sequence.append({
                'item_id': item_id,
                'score': score,
                'image_path': image_path  # Contains image path
            })
        
        sample = {
            'user_id': user_id,
            'interaction_sequence': interaction_sequence,  # Sequence containing item_id, score, and image_path
            'num_interactions': len(interaction_sequence)
        }
        
        samples.append(sample)
    
    print(f"  - Created {len(samples)} samples")
    avg_interactions = sum(s['num_interactions'] for s in samples) / len(samples) if samples else 0
    print(f"  - Average interactions per user: {avg_interactions:.1f}")
    print(f"  - Found images: {found_images:,}")
    print(f"  - Missing images: {missing_images:,}")
    if found_images + missing_images > 0:
        print(f"  - Image coverage: {found_images/(found_images+missing_images)*100:.1f}%")
    
    return samples


def split_dataset(samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/val/test (by interactions)"""
    print("\n" + "=" * 80)
    print("Splitting dataset by interactions (Interaction Split)...")
    print("=" * 80)
    
    # Set random seed to ensure split consistency
    random.seed(RANDOM_SEED)
    
    # Split by interactions: split each user's interactions
    train_samples = []
    val_samples = []
    test_samples = []
    
    train_interactions_count = 0
    val_interactions_count = 0
    test_interactions_count = 0
    
    for user_entry in samples:
        user_id = user_entry['user_id']
        interactions = user_entry['interaction_sequence'].copy()  # Copy list
        
        # If interactions are too few (e.g., less than 5), put all in training set to prevent insufficient validation/test data
        if len(interactions) < 5:
            train_samples.append({
                "user_id": user_id,
                "interaction_sequence": interactions,
                "num_interactions": len(interactions)
            })
            train_interactions_count += len(interactions)
            continue
        
        # Shuffle this user's interaction records
        random.shuffle(interactions)
        
        # 70% training, 15% validation, 15% test
        total = len(interactions)
        train_split = int(total * TRAIN_RATIO)
        val_split = int(total * (TRAIN_RATIO + VAL_RATIO))
        
        train_interactions = interactions[:train_split]
        val_interactions = interactions[train_split:val_split]
        test_interactions = interactions[val_split:]
        
        # Build new entry
        if train_interactions:
            train_samples.append({
                "user_id": user_id,
                "interaction_sequence": train_interactions,
                "num_interactions": len(train_interactions)
            })
            train_interactions_count += len(train_interactions)
        
        if val_interactions:
            val_samples.append({
                "user_id": user_id,
                "interaction_sequence": val_interactions,
                "num_interactions": len(val_interactions)
            })
            val_interactions_count += len(val_interactions)
        
        if test_interactions:
            test_samples.append({
                "user_id": user_id,
                "interaction_sequence": test_interactions,
                "num_interactions": len(test_interactions)
            })
            test_interactions_count += len(test_interactions)
    
    total_interactions = train_interactions_count + val_interactions_count + test_interactions_count
    
    print(f"  - Total users: {len(samples)}")
    print(f"  - Train users: {len(train_samples)}")
    print(f"  - Val users: {len(val_samples)}")
    print(f"  - Test users: {len(test_samples)}")
    
    print(f"\n  - Total interactions: {total_interactions:,}")
    print(f"  - Train interactions: {train_interactions_count:,} ({train_interactions_count/total_interactions*100:.1f}%)")
    print(f"  - Val interactions: {val_interactions_count:,} ({val_interactions_count/total_interactions*100:.1f}%)")
    print(f"  - Test interactions: {test_interactions_count:,} ({test_interactions_count/total_interactions*100:.1f}%)")
    
    # Check user overlap
    train_users = set(entry['user_id'] for entry in train_samples)
    val_users = set(entry['user_id'] for entry in val_samples)
    test_users = set(entry['user_id'] for entry in test_samples)
    
    print("\n  - User overlap check:")
    print(f"    Train-Val overlap: {len(train_users & val_users)} users")
    print(f"    Train-Test overlap: {len(train_users & test_users)} users")
    print(f"    Val-Test overlap: {len(val_users & test_users)} users")
    
    if len(train_users & test_users) == 0:
        print("\n  ⚠️  Warning: No user overlap between train and test sets!")
    else:
        print("\n  ✓ Train and test sets have overlapping users, as required!")
    
    return train_samples, val_samples, test_samples


def save_datasets(train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
    """Save datasets to JSON files"""
    print("\n" + "=" * 80)
    print("Saving datasets...")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save training set
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    print(f"  - Saved train set: {OUTPUT_TRAIN} ({len(train_samples)} samples)")
    
    # Save validation set
    with open(OUTPUT_VAL, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    print(f"  - Saved val set: {OUTPUT_VAL} ({len(val_samples)} samples)")
    
    # Save test set
    with open(OUTPUT_TEST, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)
    print(f"  - Saved test set: {OUTPUT_TEST} ({len(test_samples)} samples)")


def print_statistics(train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
    """Print statistics"""
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
        print(f"  - First 3 interactions:")
        for i, interaction in enumerate(sample['interaction_sequence'][:3]):
            image_path = interaction.get('image_path', 'None')
            print(f"      [{i+1}] item_id: {interaction['item_id']}, score: {interaction['score']}, image_path: {image_path}")


def main():
    """Main function"""
    print("=" * 80)
    print("FLICKR-AES Dataset Processing (Interaction Split)")
    print("=" * 80)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Split ratios (by interactions): Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    print("Note: This ensures train and test sets have overlapping users.")
    
    # 1. Load user rating data
    df = load_worker_data()
    
    # 2. Create user interaction sequences
    user_interactions = create_user_interaction_sequences(df)
    
    # 3. Create dataset samples
    samples = create_dataset_samples(user_interactions)
    
    # 4. Split dataset
    train_samples, val_samples, test_samples = split_dataset(samples)
    
    # 5. Save datasets
    save_datasets(train_samples, val_samples, test_samples)
    
    # 6. Print statistics
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

