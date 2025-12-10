#!/usr/bin/env python3
"""
Train Personalized Verifier
Using Neural Matrix Factorization (NCF) architecture
"""

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

# ================= Configuration Area =================
CONFIG = {
    "embedding_path": "{FLICKR_AES_BASE_PATH}/embeddings/image_embeddings.npy",
    "image_paths_path": "{FLICKR_AES_BASE_PATH}/embeddings/image_paths.json",
    "train_json": "{FLICKR_AES_BASE_PATH}/processed_dataset/resplit/train.json",
    "val_json": "{FLICKR_AES_BASE_PATH}/processed_dataset/resplit/val.json",
    "input_dim": 1024,   # ViT-H-14 dimension
    "user_emb_dim": 64,  # User embedding dimension
    "hidden_dim": 256,   # MLP hidden layer
    "batch_size": 128,
    "lr": 1e-4,
    "epochs": 20,
    "weight_decay": 1e-5,
    "dropout": 0.2,
    "threshold": 4.0  # Score >= 4.0 is positive
}

# ==========================================

class VerifierNet(nn.Module):
    """
    Neural Matrix Factorization (NMF) style verifier
    Corresponds to paper formula: v(U, I) = g(U, CLIP(I))
    """
    def __init__(self, num_users, input_img_dim, user_emb_dim, hidden_dim, dropout=0.2):
        super().__init__()
        
        # 1. User side: Embedding
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        
        # 2. Image side: CLIP feature projection (optional, can also use MLP only)
        # Maps CLIP high-dimensional features to a space suitable for user interaction
        self.img_projector = nn.Sequential(
            nn.Linear(input_img_dim, input_img_dim), 
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. Interaction and prediction MLP g_gamma
        # Input is concatenation of User_Emb (64) + Image_Feat (1024)
        self.classifier = nn.Sequential(
            nn.Linear(user_emb_dim + input_img_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # Output probability P(Like | U, I)
        )
    
    def forward(self, user_idx, img_feats):
        # Get user vector
        u_emb = self.user_embedding(user_idx) # [B, 64]
        
        # Process image vector (img_feats is already CLIP embedding)
        i_feat = self.img_projector(img_feats) # [B, 1024]
        
        # Concatenate
        combined = torch.cat([u_emb, i_feat], dim=1) # [B, 1088]
        
        # Predict
        score = self.classifier(combined)
        return score.squeeze()


class FlickrAESDataset(Dataset):
    def __init__(self, json_path, npy_path, image_paths_path, threshold=4.0):
        """
        Initialize dataset
        
        Args:
            json_path: Path to training/validation JSON file
            npy_path: Path to .npy file containing CLIP embeddings
            image_paths_path: JSON file containing list of image paths
            threshold: Score threshold, >= threshold is positive (1), otherwise negative (0)
        """
        print(f"Loading image embeddings from {npy_path}...")
        # 1. Load Embeddings
        self.img_embeddings = np.load(npy_path).astype(np.float32)
        print(f"Embeddings shape: {self.img_embeddings.shape}")
        
        # 2. Build filename -> .npy index mapping
        print(f"Building filename to index mapping...")
        with open(image_paths_path, 'r') as f:
            file_paths = json.load(f)  # Full path list
        
        self.file_to_idx = {}
        for idx, full_path in enumerate(file_paths):
            # Extract filename from full path
            filename = Path(full_path).name
            self.file_to_idx[filename] = idx
        
        print(f"Found {len(self.file_to_idx)} image file mappings")
        
        # 3. Parse User JSON and flatten data
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r') as f:
            raw_data = json.load(f)  # list of dicts
            
        self.samples = []
        self.user_to_id = {}  # Map string user_id to 0,1,2...
        
        current_u_idx = 0
        matched_count = 0
        unmatched_count = 0
        unmatched_items = []  # Record unmatched item_ids (for debugging)
        
        for entry in raw_data:
            u_str = entry['user_id']
            if u_str not in self.user_to_id:
                self.user_to_id[u_str] = current_u_idx
                current_u_idx += 1
            
            u_idx = self.user_to_id[u_str]
            
            for interaction in entry['interaction_sequence']:
                item_id = interaction['item_id']
                score = interaction['score']
                
                # Only add to training if we can find this image in .npy
                if item_id in self.file_to_idx:
                    img_idx = self.file_to_idx[item_id]
                    # ★★★ Core: Binarize Label ★★★
                    # Paper setting: Rating >= threshold is 1 (Like), otherwise 0
                    label = 1.0 if score >= threshold else 0.0
                    self.samples.append((u_idx, img_idx, label))
                    matched_count += 1
                else:
                    unmatched_count += 1
                    # Record first 10 unmatched item_ids for debugging
                    if len(unmatched_items) < 10:
                        unmatched_items.append(item_id)
        
        print(f"Total users: {len(self.user_to_id)}")
        print(f"Total samples: {len(self.samples)}")
        print(f"Matched: {matched_count}, Unmatched: {unmatched_count}")
        
        # Debug info: show unmatched examples
        if unmatched_count > 0:
            for item_id in unmatched_items:
                print(f"  - {item_id}")
            if unmatched_count > 10:
                print(f"  ... {unmatched_count - 10} more unmatched images")
        
        # Statistics of positive/negative samples
        if len(self.samples) > 0:
            positive_count = sum(1 for _, _, label in self.samples if label == 1.0)
            negative_count = len(self.samples) - positive_count
            print(f"Positive samples (>= {threshold}): {positive_count} ({positive_count/len(self.samples)*100:.2f}%)")
            print(f"Negative samples (< {threshold}): {negative_count} ({negative_count/len(self.samples)*100:.2f}%)")
        else:
            print("[WARNING] Dataset is empty, no valid samples!")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        u_idx, img_row_idx, label = self.samples[idx]
        
        # Safety check: ensure index is within valid range
        if img_row_idx >= len(self.img_embeddings):
            print(f"[WARNING] Image index out of bounds: {img_row_idx} >= {len(self.img_embeddings)}, skipping this sample")
            # Return an invalid sample, will be filtered later
            return None
        
        # Read image features from in-memory large matrix
        try:
            img_vec = self.img_embeddings[img_row_idx]
        except IndexError as e:
            print(f"[WARNING] Cannot read image features, index: {img_row_idx}, error: {e}")
            return None
        
        return {
            "user": torch.tensor(u_idx, dtype=torch.long),
            "image": torch.tensor(img_vec, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32)
        }


def evaluate_model(model, dataloader, device, criterion):
    """Evaluate model"""
    # Check if dataloader is empty
    if len(dataloader) == 0:
        print("Warning: DataLoader is empty, cannot evaluate")
        return {
            'loss': float('inf'),
            'accuracy': 0.0,
            'auc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            u = batch['user'].to(device)
            i = batch['image'].to(device)
            y = batch['label'].to(device)
            
            preds = model(u, i)
            loss = criterion(preds, y)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # Check if there are any predictions
    if len(all_preds) == 0:
        print("Warning: No predictions collected")
        return {
            'loss': float('inf'),
            'accuracy': 0.0,
            'auc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_labels = (all_preds >= 0.5).astype(int)
    
    accuracy = accuracy_score(all_labels, pred_labels)
    try:
        # Calculate ROC-AUC (core metric used in paper)
        # roc_auc_score calculates the Area Under ROC Curve
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError as e:
        print(f"Warning: Cannot calculate ROC-AUC: {e}")
        auc = 0.0  # If only one class (cannot calculate AUC)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, pred_labels, average='binary', zero_division=0
    )
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_verifier(config):
    """Train verifier model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare data
    print("=" * 60)
    
    # First collect all user IDs (from training and validation sets)
    print("Collecting all user IDs...")
    with open(config['train_json'], 'r') as f:
        train_raw_data = json.load(f)
    with open(config['val_json'], 'r') as f:
        val_raw_data = json.load(f)
    
    all_user_ids = set()
    for entry in train_raw_data:
        all_user_ids.add(entry['user_id'])
    for entry in val_raw_data:
        all_user_ids.add(entry['user_id'])
    
    # Create unified user mapping
    unified_user_map = {uid: idx for idx, uid in enumerate(sorted(all_user_ids))}
    print(f"Unified user mapping: {len(unified_user_map)} users")
    
    # Load training data
    print("\nLoading training data...")
    train_dataset = FlickrAESDataset(
        config['train_json'], 
        config['embedding_path'], 
        config['image_paths_path'],
        threshold=config['threshold']
    )
    
    # Remap training set user IDs
    train_samples = []
    for u_idx, img_idx, label in train_dataset.samples:
        # Find original user ID
        original_uid = None
        for uid, old_idx in train_dataset.user_to_id.items():
            if old_idx == u_idx:
                original_uid = uid
                break
        if original_uid and original_uid in unified_user_map:
            new_u_idx = unified_user_map[original_uid]
            train_samples.append((new_u_idx, img_idx, label))
    train_dataset.samples = train_samples
    train_dataset.user_to_id = unified_user_map
    
    # Load validation data
    print("\nLoading validation data...")
    val_dataset = FlickrAESDataset(
        config['val_json'], 
        config['embedding_path'], 
        config['image_paths_path'],
        threshold=config['threshold']
    )
    
    # Remap validation set user IDs
    val_samples = []
    for u_idx, img_idx, label in val_dataset.samples:
        # Find original user ID
        original_uid = None
        for uid, old_idx in val_dataset.user_to_id.items():
            if old_idx == u_idx:
                original_uid = uid
                break
        if original_uid and original_uid in unified_user_map:
            new_u_idx = unified_user_map[original_uid]
            val_samples.append((new_u_idx, img_idx, label))
    val_dataset.samples = val_samples
    val_dataset.user_to_id = unified_user_map
    
    num_users = len(unified_user_map)
    print(f"\nUnified user count: {num_users}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError("Training set is empty! Please check data paths and image mappings.")
    if len(val_dataset) == 0:
        raise ValueError("Validation set is empty! Please check data paths and image mappings.")
    
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 2. Initialize model
    print("\n" + "=" * 60)
    print("Initializing model...")
    model = VerifierNet(
        num_users=num_users,
        input_img_dim=config['input_dim'],
        user_emb_dim=config['user_emb_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(device)
    
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    criterion = nn.BCELoss()  # Binary cross entropy
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 4. Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_val_auc = 0.0
    best_epoch = 0
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in progress_bar:
            u = batch['user'].to(device)
            i = batch['image'].to(device)
            y = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            preds = model(u, i)
            loss = criterion(preds, y)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Simple accuracy calculation
            predicted_labels = (preds >= 0.5).float()
            correct += (predicted_labels == y).sum().item()
            total += y.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Print results
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"ROC-AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Save best model (based on validation set ROC-AUC)
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch + 1
            print(f"  ✓ New best model (ROC-AUC: {best_val_auc:.4f})")
            
            # Save model
            output_dir = Path(config.get('output_dir', './checkpoints'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'val_metrics': val_metrics,
                'config': config,
                'num_users': num_users
            }, output_dir / 'best_model.pth')
            
            # Save user mapping
            with open(output_dir / 'user_map.json', 'w') as f:
                json.dump(unified_user_map, f, indent=2)
        
        print("-" * 60)
    
    print(f"\nTraining completed! Best model at Epoch {best_epoch}, Validation ROC-AUC: {best_val_auc:.4f}")
    return model, unified_user_map, best_epoch


def evaluate_on_testset(config, model_path, user_map_path, auc_threshold=0.87):
    """Evaluate model on test set"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("Evaluating model on test set")
    print("=" * 60)
    
    # Load user mapping
    with open(user_map_path, 'r') as f:
        user_map = json.load(f)
    num_users = len(user_map)
    
    # Load test data
    print("Loading test data...")
    test_dataset = FlickrAESDataset(
        config['test_json'], 
        config['embedding_path'], 
        config['image_paths_path'],
        threshold=config['threshold']
    )
    
    # Remap test set user IDs
    test_samples = []
    skipped_count = 0
    skipped_users = []  # Record skipped users (for debugging)
    skipped_user_count = {}  # Count skipped samples per user
    
    for u_idx, img_idx, label in test_dataset.samples:
        # Find original user ID
        original_uid = None
        for uid, old_idx in test_dataset.user_to_id.items():
            if old_idx == u_idx:
                original_uid = uid
                break
        if original_uid and original_uid in user_map:
            new_u_idx = user_map[original_uid]
            test_samples.append((new_u_idx, img_idx, label))
        else:
            # If test set has new users, skip this sample
            skipped_count += 1
            if original_uid:
                if original_uid not in skipped_user_count:
                    skipped_user_count[original_uid] = 0
                    if len(skipped_users) < 10:
                        skipped_users.append(original_uid)
                skipped_user_count[original_uid] += 1
            continue
    
    test_dataset.samples = test_samples
    test_dataset.user_to_id = user_map
    
    print(f"Original test samples: {len(test_dataset.samples) + skipped_count}")
    print(f"Valid test samples: {len(test_dataset)}")
    if skipped_count > 0:
        print(f"Skipped samples (new users): {skipped_count}")
        for uid in skipped_users[:10]:
            count = skipped_user_count.get(uid, 0)
            print(f"  - User {uid}: skipped {count} samples")
        if len(skipped_users) > 10:
            print(f"  ... {len(skipped_users) - 10} more users skipped")
    
    # Check if there are valid samples
    if len(test_dataset) == 0:
        print("\nError: No valid samples in test set!")
        print("Possible reasons:")
        print("  1. All users in test set are not in training set")
        print("  2. All images in test set are not in embeddings")
        print("  3. Test set itself is empty")
        return None, False
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = VerifierNet(
        num_users=num_users,
        input_img_dim=config['input_dim'],
        user_emb_dim=config['user_emb_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate
    criterion = nn.BCELoss()
    test_metrics = evaluate_model(model, test_loader, device, criterion)
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Set Evaluation Results")
    print("=" * 60)
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {test_metrics['auc']:.4f}  <-- Core metric used in paper")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print("=" * 60)
    
    # Check if threshold is reached
    test_auc = test_metrics['auc']  # This is ROC-AUC
    if test_auc >= auc_threshold:
        print(f"\n✓ Verifier passed test! ROC-AUC ({test_auc:.4f}) >= threshold ({auc_threshold:.4f})")
        print("  Verifier is qualified to score generated models.")
        qualified = True
    else:
        print(f"\n✗ Verifier failed test. ROC-AUC ({test_auc:.4f}) < threshold ({auc_threshold:.4f})")
        print("  Verifier needs further training or tuning to score generated models.")
        qualified = False
    
    # Save test results
    output_dir = Path(config['output_dir'])
    test_results = {
        'test_metrics': test_metrics,
        'auc_threshold': auc_threshold,
        'qualified': qualified,
        'model_path': str(model_path),
        'test_json': config['test_json']
    }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest results saved to: {output_dir / 'test_results.json'}")
    
    return test_metrics, qualified


def main():
    parser = argparse.ArgumentParser(description='Train Personalized Verifier')
    parser.add_argument('--embedding_path', type=str, default=CONFIG['embedding_path'])
    parser.add_argument('--image_paths_path', type=str, default=CONFIG['image_paths_path'])
    parser.add_argument('--train_json', type=str, default=CONFIG['train_json'])
    parser.add_argument('--val_json', type=str, default=CONFIG['val_json'])
    parser.add_argument('--test_json', type=str, 
                       default='{FLICKR_AES_BASE_PATH}/processed_dataset/resplit/test.json')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=CONFIG['lr'])
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--user_emb_dim', type=int, default=CONFIG['user_emb_dim'])
    parser.add_argument('--hidden_dim', type=int, default=CONFIG['hidden_dim'])
    parser.add_argument('--threshold', type=float, default=CONFIG['threshold'])
    parser.add_argument('--dropout', type=float, default=CONFIG['dropout'])
    parser.add_argument('--weight_decay', type=float, default=CONFIG['weight_decay'])
    parser.add_argument('--auc_threshold', type=float, default=0.87,
                       help='Test set ROC-AUC threshold, verifier is qualified only if this value is reached (paper uses ~0.87)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, only evaluate on test set (requires trained model)')
    
    args = parser.parse_args()
    
    config = {
        'embedding_path': args.embedding_path,
        'image_paths_path': args.image_paths_path,
        'train_json': args.train_json,
        'val_json': args.val_json,
        'test_json': args.test_json,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'input_dim': CONFIG['input_dim'],
        'user_emb_dim': args.user_emb_dim,
        'hidden_dim': args.hidden_dim,
        'threshold': args.threshold,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay
    }
    
    print("=" * 60)
    print("Personalized Verifier Training and Evaluation")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  auc_threshold: {args.auc_threshold}")
    print("=" * 60)
    
    if not args.skip_training:
        # Train model
        model, user_map, best_epoch = train_verifier(config)
        print("\nTraining completed!")
        
        # Load best model for testing
        model_path = Path(config['output_dir']) / 'best_model.pth'
        user_map_path = Path(config['output_dir']) / 'user_map.json'
    else:
        # Only evaluate
        model_path = Path(config['output_dir']) / 'best_model.pth'
        user_map_path = Path(config['output_dir']) / 'user_map.json'
        
        if not model_path.exists():
            print(f"Error: Model file does not exist: {model_path}")
            return
        if not user_map_path.exists():
            print(f"Error: User map file does not exist: {user_map_path}")
            return
        
        # Load config from checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'config' in checkpoint:
            # Update some values in config
            for key in ['input_dim', 'user_emb_dim', 'hidden_dim', 'dropout', 'threshold']:
                if key in checkpoint['config']:
                    config[key] = checkpoint['config'][key]
    
    # Evaluate on test set
    test_metrics, qualified = evaluate_on_testset(
        config, 
        model_path, 
        user_map_path,
        auc_threshold=args.auc_threshold
    )
    
    if qualified:
        print("\n✓ Verifier is ready to score generated models!")
    else:
        print("\n✗ Verifier needs further improvement.")


if __name__ == "__main__":
    main()

