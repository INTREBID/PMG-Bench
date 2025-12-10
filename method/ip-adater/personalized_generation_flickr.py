
import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import argparse
from datetime import datetime
import random

import sys

from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms

# Metrics imports
import lpips
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import clip

# HPSv2
HPSv2_AVAILABLE = False
hpsv2_module = None
try:
    import hpsv2
    hpsv2_module = hpsv2
    HPSv2_AVAILABLE = True
except ImportError:
    try:
        from hpsv2 import HPSv2
        HPSv2_AVAILABLE = True
    except ImportError:
        pass

# LAION Aesthetic
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class MetricsEvaluator:
    """Metrics evaluator for generated images"""

    def __init__(self, device: str = "cuda", image_size: int = 512, styles_path: Optional[str] = None):
        self.device = torch.device(device)
        self.image_size = image_size
        
        self.user_styles = {}
        if styles_path:
            self._load_user_styles(styles_path)

        print("Initializing metrics...")
        self._init_metrics()
    
    def _load_user_styles(self, styles_path: str):
        """Load user style preferences"""
        try:
            if os.path.exists(styles_path):
                with open(styles_path, 'r', encoding='utf-8') as f:
                    styles_data = json.load(f)
                for item in styles_data:
                    worker_id = item.get('worker', '')
                    style_text = item.get('style', '')
                    if worker_id and style_text:
                        self.user_styles[worker_id] = style_text
                print(f"[DEBUG] Loaded styles for {len(self.user_styles)} users from {styles_path}")
            else:
                print(f"[WARNING] Styles file not found: {styles_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load user styles: {e}")
            self.user_styles = {}
    
    def _get_user_preference_text(self, user_id: str) -> Optional[str]:
        """Get user preference text from user_id"""
        if user_id in self.user_styles:
            style_text = self.user_styles[user_id]
            if "Style preferences:" in style_text:
                style_text = style_text.split("Style preferences:")[-1]
            lines = style_text.split('\n')
            keywords = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('*'):
                    continue
                if ',' in line:
                    keywords.extend([k.strip() for k in line.split(',') if k.strip()])
                else:
                    keywords.extend([k.strip() for k in line.split() if k.strip()])
            unique_keywords = list(dict.fromkeys(keywords))
            preference_text = ", ".join(unique_keywords[:20])
            return preference_text if preference_text else None
        return None

    def _init_metrics(self):
        """Initialize all evaluation metrics"""
        # LPIPS
        try:
            self.lpips_metric = lpips.LPIPS(net='vgg').to(self.device)
            self.lpips_metric.eval()
            print("LPIPS initialized")
        except Exception as e:
            print(f"Error initializing LPIPS: {e}")
            self.lpips_metric = None

        # SSIM
        try:
            self.ssim_metric = SSIM(data_range=2.0).to(self.device)
            print("SSIM initialized")
        except Exception as e:
            print(f"Error initializing SSIM: {e}")
            self.ssim_metric = None

        # CLIP
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            self.clip_transform = transforms.Compose([
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                   std=(0.26862954, 0.26130258, 0.27577711))
            ])
            print("CLIP initialized")
        except Exception as e:
            print(f"Error initializing CLIP: {e}")
            self.clip_model = None

        # HPSv2
        self.hpsv2_model = None
        self.hpsv2_failed = False
        if HPSv2_AVAILABLE and hpsv2_module is not None:
            try:
                if hasattr(hpsv2_module, 'score'):
                    self.hpsv2_model = hpsv2_module
                    print("HPSv2 initialized (module mode)")
                else:
                    try:
                        self.hpsv2_model = hpsv2_module.HPSv2(device=self.device)
                        print("HPSv2 initialized (class mode)")
                    except:
                        self.hpsv2_model = hpsv2_module.HPSv2()
                        if hasattr(self.hpsv2_model, 'to'):
                            self.hpsv2_model = self.hpsv2_model.to(self.device)
                        print("HPSv2 initialized (class mode, no device)")
            except Exception as e:
                print(f"Warning: HPSv2 initialization failed: {e}")
                self.hpsv2_failed = True

        self.laion_aesthetic_model = None
        self._laion_loaded = False
        print("Metrics initialization complete")

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor [0, 1] (matching PMG_Repro)"""
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor

    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """Calculate LPIPS"""
        if self.lpips_metric is None:
            print("[DEBUG] LPIPS metric not available, skipping calculation")
            return None
        try:
            img1_norm = (img1 * 2.0) - 1.0
            img2_norm = (img2 * 2.0) - 1.0
            with torch.no_grad():
                score = self.lpips_metric(img1_norm, img2_norm)
            score_value = score.item()
            print(f"[DEBUG] LPIPS calculated: {score_value:.4f}")
            return score_value
        except Exception as e:
            print(f"[WARNING] Error calculating LPIPS: {e}")
            return None

    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """Calculate SSIM"""
        if self.ssim_metric is None:
            print("[DEBUG] SSIM metric not available, skipping calculation")
            return None
        try:
            with torch.no_grad():
                score = self.ssim_metric(img1, img2)
            score_value = score.item()
            print(f"[DEBUG] SSIM calculated: {score_value:.4f}")
            return score_value
        except Exception as e:
            print(f"[WARNING] Error calculating SSIM: {e}")
            return None

    def calculate_cps(self, image_tensor: torch.Tensor, user_id: str, fallback_text: Optional[str] = None) -> Optional[float]:
        """Calculate CPS: CLIP image-text similarity"""
        if self.clip_model is None:
            print("[DEBUG] CLIP model not available, skipping CPS calculation")
            return None
        preference_text = self._get_user_preference_text(user_id)
        if preference_text is None:
            if fallback_text:
                preference_text = fallback_text
                print(f"[DEBUG] User preference not found for {user_id}, using fallback text")
            else:
                print(f"[DEBUG] No preference text available for user {user_id}")
                return None
        
        try:
            with torch.no_grad():
                image_clip = self.clip_transform(image_tensor)
                image_features = self.clip_model.encode_image(image_clip)
                text_tokens = clip.tokenize([preference_text]).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).item()
            print(f"[DEBUG] CPS calculated: {similarity:.4f} (user: {user_id}, text: {preference_text[:50]}...)")
            return similarity
        except Exception as e:
            print(f"[WARNING] Error calculating CPS: {e}")
            return None

    def calculate_cpis(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """Calculate CPIS: CLIP image-image similarity"""
        if self.clip_model is None:
            print("[DEBUG] CLIP model not available, skipping CPIS calculation")
            return None
        try:
            with torch.no_grad():
                img1_clip = self.clip_transform(img1)
                img2_clip = self.clip_transform(img2)
                features1 = self.clip_model.encode_image(img1_clip)
                features2 = self.clip_model.encode_image(img2_clip)
                features1 = features1 / features1.norm(dim=-1, keepdim=True)
                features2 = features2 / features2.norm(dim=-1, keepdim=True)
                similarity = (features1 @ features2.T).item()
            print(f"[DEBUG] CPIS calculated: {similarity:.4f}")
            return similarity
        except Exception as e:
            print(f"[WARNING] Error calculating CPIS: {e}")
            return None

    def calculate_hpsv2(self, image_path: str, prompt: str) -> Optional[float]:
        """Calculate HPSv2"""
        if self.hpsv2_model is None or self.hpsv2_failed:
            print("[DEBUG] HPSv2 model not available or failed, skipping calculation")
            return None
        try:
            print(f"[DEBUG] Calculating HPSv2 for image: {image_path}")
            if hasattr(self.hpsv2_model, 'score') and callable(self.hpsv2_model.score):
                try:
                    result = self.hpsv2_model.score(image_path, prompt, hps_version="v2.1")
                except TypeError:
                    try:
                        result = self.hpsv2_model.score([image_path], [prompt], hps_version="v2.1")
                    except:
                        result = self.hpsv2_model.score([image_path], prompt, hps_version="v2.1")
            else:
                result = self.hpsv2_model.score(image_path, prompt, hps_version="v2.1")

            if isinstance(result, (list, tuple)):
                score_value = float(result[0])
            elif isinstance(result, torch.Tensor):
                score_value = float(result.item())
            elif isinstance(result, np.ndarray):
                score_value = float(result.item() if result.size == 1 else result[0])
            else:
                score_value = float(result)
            
            print(f"[DEBUG] HPSv2 calculated: {score_value:.4f}")
            return score_value
        except Exception as e:
            if not self.hpsv2_failed:
                print(f"[WARNING] HPSv2 calculation failed: {e}")
                self.hpsv2_failed = True
            return None

    def calculate_laion_aesthetic(self, image: Image.Image) -> Optional[float]:
        """
        Calculate LAION aesthetic score using CLIP + Linear Regression Head
        
        Process:
        1. Extract image features using CLIP (ViT-B/32, 512 dim)
        2. Predict aesthetic score using linear regression head
        """
        if not self._laion_loaded:
            print("Loading LAION aesthetic predictor (linear head)...")
            
            model_path = "{LAION_AESTHETIC_MODEL_PATH}"
            
            if not os.path.exists(model_path):
                print(f"Warning: LAION aesthetic model not found at {model_path}")
                return None
            
            try:
                import torch.nn as nn
                self.laion_aesthetic_model = nn.Linear(512, 1)
                
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                self.laion_aesthetic_model.load_state_dict(state_dict)
                self.laion_aesthetic_model.eval()
                self.laion_aesthetic_model = self.laion_aesthetic_model.to(self.device).float()
                
                self._laion_loaded = True
                print("LAION aesthetic predictor loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load LAION aesthetic model: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        if self.laion_aesthetic_model is None:
            return None
        
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                
                image_features = image_features.float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                norm_value = image_features.norm(dim=-1).item()
                if abs(norm_value - 1.0) > 0.01:
                    print(f"[ERROR] LAION normalization failed! Norm value: {norm_value:.6f} (expected ~1.0)")
                    print(f"[ERROR] This will cause score explosion. Stopping immediately.")
                    raise ValueError(f"Normalization failed: norm={norm_value:.6f}")
                
                aesthetic_score = self.laion_aesthetic_model(image_features)
                score = aesthetic_score.item()
                
                if score >= 10.0:
                    print(f"[ERROR] LAION aesthetic score is too high: {score:.4f} (expected range: 0-10)")
                    print(f"[ERROR] Feature norm after normalization: {norm_value:.6f}")
                    print(f"[ERROR] Feature dtype: {image_features.dtype}")
                    print(f"[ERROR] Feature shape: {image_features.shape}")
                    print(f"[ERROR] Feature range: [{image_features.min().item():.4f}, {image_features.max().item():.4f}]")
                    raise ValueError(f"LAION score {score:.4f} exceeds normal range (0-10). Calculation error detected!")
            
            return float(score)
        except Exception as e:
            print(f"[ERROR] LAION aesthetic calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_generated_image(
        self,
        generated_image: Image.Image,
        target_image_path: str,
        history_images: List[Image.Image],
        target_caption: str,
        generated_image_path: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Optional[float]]:
        """Evaluate generated image and return all metrics"""
        print(f"[DEBUG] Starting evaluation for generated image: {generated_image_path}")
        print(f"[DEBUG]   Target image: {target_image_path}")
        print(f"[DEBUG]   Number of history images: {len(history_images)}")
        print(f"[DEBUG]   Target caption: {target_caption[:100]}..." if len(target_caption) > 100 else f"[DEBUG]   Target caption: {target_caption}")
        
        metrics = {}

        print("[DEBUG] Converting generated image to tensor...")
        generated_tensor = self._image_to_tensor(generated_image)
        print(f"[DEBUG]   Generated image tensor shape: {generated_tensor.shape}")
        try:
            print(f"[DEBUG] Loading target image from: {target_image_path}")
            target_image = Image.open(target_image_path).convert("RGB")
            target_tensor = self._image_to_tensor(target_image)
            print(f"[DEBUG]   Target image loaded successfully, shape: {target_tensor.shape}")
        except Exception as e:
            print(f"[WARNING] Failed to load target image: {e}")
            target_tensor = None

        # LPIPS vs Target
        if target_tensor is not None:
            print("[DEBUG] Calculating LPIPS vs Target...")
            metrics['lpips_target'] = self.calculate_lpips(generated_tensor, target_tensor)
            print("[DEBUG] Calculating SSIM vs Target...")
            metrics['ssim_target'] = self.calculate_ssim(generated_tensor, target_tensor)
        else:
            print("[DEBUG] Skipping target comparison metrics (target image not available)")
            metrics['lpips_target'] = None
            metrics['ssim_target'] = None

        # LPIPS vs History Avg, SSIM vs History Avg, CPIS vs History Avg
        print(f"[DEBUG] Calculating metrics vs {len(history_images)} history images...")
        lpips_history_scores = []
        ssim_history_scores = []
        cpis_history_scores = []

        for i, hist_img in enumerate(history_images):
            try:
                print(f"[DEBUG]   Processing history image {i+1}/{len(history_images)}...")
                hist_tensor = self._image_to_tensor(hist_img)
                lpips_hist = self.calculate_lpips(generated_tensor, hist_tensor)
                ssim_hist = self.calculate_ssim(generated_tensor, hist_tensor)
                cpis_hist = self.calculate_cpis(generated_tensor, hist_tensor)

                if lpips_hist is not None:
                    lpips_history_scores.append(lpips_hist)
                if ssim_hist is not None:
                    ssim_history_scores.append(ssim_hist)
                if cpis_hist is not None:
                    cpis_history_scores.append(cpis_hist)
            except Exception as e:
                print(f"[WARNING] Failed to process history image {i+1}: {e}")
                continue

        metrics['lpips_history_avg'] = np.mean(lpips_history_scores) if lpips_history_scores else None
        metrics['ssim_history_avg'] = np.mean(ssim_history_scores) if ssim_history_scores else None
        metrics['cpis_history_avg'] = np.mean(cpis_history_scores) if cpis_history_scores else None
        
        lpips_str = f"{metrics['lpips_history_avg']:.4f}" if metrics['lpips_history_avg'] is not None else 'N/A'
        ssim_str = f"{metrics['ssim_history_avg']:.4f}" if metrics['ssim_history_avg'] is not None else 'N/A'
        cpis_str = f"{metrics['cpis_history_avg']:.4f}" if metrics['cpis_history_avg'] is not None else 'N/A'
        print(f"[DEBUG] History averages - LPIPS: {lpips_str}, SSIM: {ssim_str}, CPIS: {cpis_str}")

        print("[DEBUG] Calculating CPS (CLIP image-text similarity)...")
        if user_id:
            metrics['cps'] = self.calculate_cps(generated_tensor, user_id, fallback_text=target_caption)
        else:
            metrics['cps'] = self.calculate_cps(generated_tensor, "", fallback_text=target_caption)

        # HPSv2
        print("[DEBUG] Calculating HPSv2...")
        metrics['hpsv2'] = self.calculate_hpsv2(generated_image_path, target_caption)

        # LAION Aesthetic
        metrics['laion_aesthetic'] = self.calculate_laion_aesthetic(generated_image)

        print(f"[DEBUG] Evaluation complete. Metrics summary:")
        for key, value in metrics.items():
            if value is not None:
                print(f"[DEBUG]   {key}: {value:.4f}")
            else:
                print(f"[DEBUG]   {key}: None")
        
        return metrics


class VerifierScorer:
    """Trained verifier scorer"""

    def __init__(self, model_path: str, user_map_path: str, device: str = "cuda"):
        self.device = torch.device(device)

        with open(user_map_path, 'r') as f:
            self.user_map = json.load(f)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        num_users = len(self.user_map)
        input_dim = checkpoint['config']['input_dim']
        user_emb_dim = checkpoint['config']['user_emb_dim']
        hidden_dim = checkpoint['config']['hidden_dim']
        dropout = checkpoint['config']['dropout']

        from train_verifier import VerifierNet
        self.model = VerifierNet(
            num_users=num_users,
            input_img_dim=input_dim,
            user_emb_dim=user_emb_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.clip_model = None
        self.clip_processor = None
        self.input_dim = input_dim
        
        print(f"[DEBUG] Loading CLIP model for feature extraction (expected dim: {input_dim})...")
        try:
            if input_dim == 1024:
                if TRANSFORMERS_AVAILABLE:
                    from transformers import CLIPProcessor, CLIPModel
                    model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
                    cache_dir = os.environ.get('HF_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
                    
                    if os.path.exists(os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")):
                        self.clip_processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
                        self.clip_model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
                        self.clip_model.eval()
                        print(f"[DEBUG] ✓ CLIP ViT-H-14 loaded successfully")
                    else:
                        print(f"[WARNING] CLIP ViT-H-14 model cache not found, will try alternative")
                        self._load_alternative_clip(input_dim)
                else:
                    print(f"[WARNING] Transformers not available, trying alternative CLIP")
                    self._load_alternative_clip(input_dim)
            else:
                self._load_alternative_clip(input_dim)
        except Exception as e:
            print(f"[WARNING] Failed to load CLIP model: {e}")
            print(f"[WARNING] Trying alternative CLIP model...")
            self._load_alternative_clip(input_dim)

        print(f"Verifier model loaded from {model_path}")

    def _load_alternative_clip(self, expected_dim: int):
        """Load alternative CLIP model"""
        try:
            if expected_dim == 768:
                clip_model_name = "ViT-L/14"
            elif expected_dim == 512:
                clip_model_name = "ViT-B/32"
            else:
                clip_model_name = "ViT-L/14"
                print(f"[WARNING] Expected dim {expected_dim} not standard, using ViT-L/14 (768 dim)")
            
            print(f"[DEBUG] Loading CLIP {clip_model_name}...")
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
            self.clip_model.eval()
            self.clip_transform = transforms.Compose([
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                   std=(0.26862954, 0.26130258, 0.27577711))
            ])
            print(f"[DEBUG] ✓ CLIP {clip_model_name} loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load alternative CLIP model: {e}")
            self.clip_model = None

    def score_image(self, user_id: str, image_path: str) -> Optional[float]:
        """Score image using verifier"""
        print(f"[DEBUG] Verifier scoring image for user_id={user_id}, image_path={image_path}")
        try:
            if user_id not in self.user_map:
                print(f"[WARNING] User {user_id} not found in user map (total users: {len(self.user_map)})")
                return None

            user_idx = self.user_map[user_id]
            print(f"[DEBUG]   User {user_id} mapped to index {user_idx}")

            if self.clip_model is None:
                print(f"[ERROR] CLIP model not available for feature extraction")
                return None

            if not os.path.exists(image_path):
                print(f"[WARNING] Image file not found: {image_path}")
                return None

            print(f"[DEBUG]   Loading image...")
            image = Image.open(image_path).convert("RGB")
            print(f"[DEBUG]   Image loaded, size: {image.size}")

            print(f"[DEBUG]   Extracting CLIP features...")
            with torch.no_grad():
                if self.clip_processor is not None:
                    inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                    image_features = self.clip_model.get_image_features(**inputs)
                else:
                    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
                    image_tensor = self.clip_transform(image_tensor)
                    image_features = self.clip_model.encode_image(image_tensor)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                if image_features.shape[-1] != self.input_dim:
                    print(f"[WARNING] CLIP feature dim {image_features.shape[-1]} != expected {self.input_dim}")
                    if image_features.shape[-1] > self.input_dim:
                        image_features = image_features[:, :self.input_dim]
                        print(f"[DEBUG]   Truncated features to {self.input_dim} dims")
                    else:
                        padding = torch.zeros(
                            image_features.shape[0], 
                            self.input_dim - image_features.shape[-1],
                            device=image_features.device
                        )
                        image_features = torch.cat([image_features, padding], dim=-1)
                        print(f"[DEBUG]   Padded features to {self.input_dim} dims")
                
                print(f"[DEBUG]   CLIP features shape: {image_features.shape}")

            print(f"[DEBUG]   Running verifier inference...")
            with torch.no_grad():
                user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                score = self.model(user_tensor, image_features)
                score_value = score.item()
                print(f"[DEBUG] ✓ Verifier score calculated: {score_value:.4f}")
                return score_value

        except Exception as e:
            print(f"[ERROR] Error scoring image with verifier: {e}")
            import traceback
            traceback.print_exc()
            return None


def load_captions(captions_path: str) -> Dict[str, str]:
    """Load image captions"""
    print(f"Loading captions from {captions_path}...")
    with open(captions_path, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    print(f"Loaded {len(captions)} captions")
    return captions


def find_image_path(image_dir: str, filename: str) -> Optional[str]:
    """Find image full path by filename"""
    possible_path = os.path.join(image_dir, filename)
    if os.path.exists(possible_path):
        print(f"[DEBUG] Found image at: {possible_path}")
        return possible_path

    if not filename.lower().endswith('.jpg'):
        possible_path = os.path.join(image_dir, filename + '.jpg')
        if os.path.exists(possible_path):
            print(f"[DEBUG] Found image with .jpg extension: {possible_path}")
            return possible_path

    print(f"[DEBUG] Image not found for filename: {filename}")
    print(f"[DEBUG]   Tried path 1: {os.path.join(image_dir, filename)}")
    if not filename.lower().endswith('.jpg'):
        print(f"[DEBUG]   Tried path 2: {os.path.join(image_dir, filename + '.jpg')}")
    print(f"[DEBUG]   Image directory exists: {os.path.exists(image_dir)}")
    if os.path.exists(image_dir):
        try:
            files_in_dir = os.listdir(image_dir)[:5]
            print(f"[DEBUG]   Sample files in directory: {files_in_dir}")
        except Exception as e:
            print(f"[DEBUG]   Error listing directory: {e}")
    
    return None


def select_user_history_images(
    user_interactions: List[Dict],
    image_dir: str,
    min_score: float = 4.0,
    max_images: int = 5
) -> List[Tuple[str, str, float]]:
    """
    Select user history images (high scores as style reference)

    Args:
        user_interactions: User interaction records
        image_dir: Image directory (as fallback path)
        min_score: Minimum score threshold
        max_images: Maximum number of images to select

    Returns:
        List of (image_path, item_id, score)
    """
    high_score_items = [
        interaction
        for interaction in user_interactions
        if interaction['score'] >= min_score
    ]

    high_score_items.sort(key=lambda x: x['score'], reverse=True)

    selected_items = high_score_items[:max_images]

    history_images = []
    print(f"[DEBUG] Selecting history images: found {len(selected_items)} high-score items (min_score={min_score})")
    for interaction in selected_items:
        item_id = interaction['item_id']
        score = interaction['score']
        print(f"[DEBUG]   Looking for history image: item_id={item_id}, score={score}")
        
        image_path = None
        if 'image_path' in interaction and interaction['image_path']:
            image_path = interaction['image_path']
            if os.path.exists(image_path):
                print(f"[DEBUG]   ✓ Using image_path from data: {image_path}")
            else:
                print(f"[DEBUG]   ⚠ image_path from data does not exist: {image_path}, trying fallback")
                image_path = None
        
        if image_path is None:
            image_path = find_image_path(image_dir, item_id)
            if image_path:
                print(f"[DEBUG]   ✓ Found image via find_image_path: {image_path}")
        
        if image_path:
            history_images.append((image_path, item_id, score))
            print(f"[DEBUG]   ✓ Successfully found history image: {image_path}")
        else:
            print(f"[WARNING] Could not find image for item_id={item_id} (score={score})")
    print(f"[DEBUG] Selected {len(history_images)}/{len(selected_items)} history images successfully")

    return history_images


def select_target_for_generation(
    user_interactions: List[Dict],
    all_captions: Dict[str, str],
    image_dir: str,
    seed: int = 42
) -> Optional[Tuple[str, str, str]]:
    """
    Select target for generation

    Strategy: Select unrated images as targets
    Simplified: Randomly select an image

    Args:
        user_interactions: User interaction records
        all_captions: All image captions
        image_dir: Image directory
        seed: Random seed

    Returns:
        (image_path, item_id, caption) or None
    """
    rated_item_ids = set(interaction['item_id'] for interaction in user_interactions)
    print(f"[DEBUG] User has rated {len(rated_item_ids)} images")
    print(f"[DEBUG] Total captions available: {len(all_captions)}")

    high_score_candidates = []
    for item_id, caption in all_captions.items():
        if item_id not in rated_item_ids:
            high_score_candidates.append((item_id, caption))

    print(f"[DEBUG] Found {len(high_score_candidates)} unrated candidate images")
    if not high_score_candidates:
        print(f"[WARNING] No unrated candidate images found for target selection")
        return None

    random.seed(seed)
    selected_item_id, selected_caption = random.choice(high_score_candidates)
    print(f"[DEBUG] Selected target for generation: item_id={selected_item_id}")
    print(f"[DEBUG]   Caption: {selected_caption[:100]}..." if len(selected_caption) > 100 else f"[DEBUG]   Caption: {selected_caption}")

    image_path = find_image_path(image_dir, selected_item_id)
    if image_path:
        print(f"[DEBUG] ✓ Target image found: {image_path}")
        return (image_path, selected_item_id, selected_caption)

    print(f"[WARNING] Target image not found for item_id={selected_item_id}")
    return None


def load_images(image_paths: List[str]) -> List[Image.Image]:
    """Load list of images"""
    images = []
    print(f"[DEBUG] Loading {len(image_paths)} images...")
    for i, img_path in enumerate(image_paths):
        try:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                print(f"[DEBUG]   [{i+1}/{len(image_paths)}] ✓ Loaded: {img_path} (size: {img.size})")
            else:
                print(f"[WARNING] Image not found: {img_path}")
        except Exception as e:
            print(f"[ERROR] Error loading image {img_path}: {e}")
            import traceback
            traceback.print_exc()
    print(f"[DEBUG] Successfully loaded {len(images)}/{len(image_paths)} images")
    return images


def setup_pipeline(
    sd_model_path: str,
    ip_adapter_path: str,
    ip_adapter_weight: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16
):
    """Setup Stable Diffusion pipeline and IP-Adapter"""
    print(f"Loading Stable Diffusion model from {sd_model_path}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_path,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    print(f"Loading IP-Adapter from {ip_adapter_path}...")
    pipe.load_ip_adapter(
        ip_adapter_path,
        subfolder="models",
        weight_name=ip_adapter_weight
    )

    return pipe


def load_multiple_ip_adapters(
    pipe,
    ip_adapter_path: str,
    ip_adapter_weight: str,
    num_adapters: int
):
    """Load multiple IP-Adapter instances"""
    if num_adapters > 1:
        current_count = 0
        if hasattr(pipe.unet, 'encoder_hid_proj') and hasattr(pipe.unet.encoder_hid_proj, 'image_projection_layers'):
            current_count = len(pipe.unet.encoder_hid_proj.image_projection_layers)

        if current_count < num_adapters:
            additional_needed = num_adapters - current_count
            print(f"Loading {additional_needed} additional IP-Adapter instances (total: {num_adapters})...")
            pipe.load_ip_adapter(
                pretrained_model_name_or_path_or_dict=[ip_adapter_path] * additional_needed,
                subfolder=["models"] * additional_needed,
                weight_name=[ip_adapter_weight] * additional_needed
            )


def generate_personalized_image(
    pipe,
    history_images: List[Image.Image],
    target_caption: str,
    target_image_path: Optional[str] = None,
    prompt: Optional[str] = None,
    negative_prompt: str = "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    ip_adapter_scale: float = 0.6,
    seed: Optional[int] = None
) -> Image.Image:
    """Generate personalized image"""
    print(f"[DEBUG] generate_personalized_image called with {len(history_images)} history images")
    
    if prompt is None:
        prompt = target_caption

    ip_adapter_images = history_images

    num_loaded = 1
    if hasattr(pipe.unet, 'encoder_hid_proj') and hasattr(pipe.unet.encoder_hid_proj, 'image_projection_layers'):
        num_loaded = len(pipe.unet.encoder_hid_proj.image_projection_layers)

    print(f"[DEBUG]   IP-Adapter instances loaded: {num_loaded}, history images: {len(history_images)}")
    
    if num_loaded < len(history_images):
        ip_adapter_images = history_images[:num_loaded]
        print(f"[DEBUG]   Using first {num_loaded} history images (truncated)")
    elif num_loaded > len(history_images):
        while len(ip_adapter_images) < num_loaded:
            ip_adapter_images.append(history_images[-1])
        print(f"[DEBUG]   Duplicated last history image to match {num_loaded} IP-Adapter instances")

    if num_loaded > 1:
        pipe.set_ip_adapter_scale([ip_adapter_scale] * num_loaded)
        print(f"[DEBUG]   Set IP-Adapter scale to {ip_adapter_scale} for {num_loaded} instances")
    else:
        pipe.set_ip_adapter_scale(ip_adapter_scale)
        print(f"[DEBUG]   Set IP-Adapter scale to {ip_adapter_scale}")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        print(f"[DEBUG]   Using seed: {seed}")

    if not prompt or len(prompt.strip()) == 0:
        print(f"[WARNING] Empty prompt detected! Using default prompt.")
        prompt = "a photo"
    
    print(f"[DEBUG]   Starting generation with prompt: {prompt[:100]}..." if len(prompt) > 100 else f"[DEBUG]   Starting generation with prompt: {prompt}")

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=ip_adapter_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

    print(f"[DEBUG]   ✓ Generation complete, image size: {result.images[0].size}")
    return result.images[0]


def process_flickr_dataset(
    test_json_path: str,
    captions_path: str,
    image_dir: str,
    output_dir: str,
    sd_model_path: str,
    ip_adapter_path: str,
    ip_adapter_weight: str,
    verifier_model_path: str,
    verifier_user_map_path: str,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    start_idx: int = 0,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    ip_adapter_scale: float = 0.6,
    seed: int = 42,
    styles_path: Optional[str] = None
):
    """Process FLICKR-AES dataset"""
    os.makedirs(output_dir, exist_ok=True)
    generated_dir = os.path.join(output_dir, "generated_images")
    os.makedirs(generated_dir, exist_ok=True)

    print(f"Loading test dataset from {test_json_path}...")
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"Loading captions from {captions_path}...")
    captions = load_captions(captions_path)

    total_samples = len(test_data)
    if max_samples is not None:
        test_data = test_data[start_idx:start_idx + max_samples]
    else:
        test_data = test_data[start_idx:]

    print(f"Processing {len(test_data)} samples (starting from index {start_idx})...")

    pipe = setup_pipeline(
        sd_model_path=sd_model_path,
        ip_adapter_path=ip_adapter_path,
        ip_adapter_weight=ip_adapter_weight,
        device=device
    )

    print("Initializing metrics evaluator...")
    evaluator = MetricsEvaluator(device=device, image_size=512, styles_path=styles_path)

    print("Initializing verifier scorer...")
    verifier = VerifierScorer(verifier_model_path, verifier_user_map_path, device=device)

    results = []
    failed_samples = []
    all_metrics = {
        'lpips_target': [], 'lpips_history_avg': [], 'ssim_target': [], 'ssim_history_avg': [],
        'cps': [], 'cpis_history_avg': [], 'hpsv2': [], 'laion_aesthetic': [], 'verifier_score': []
    }

    for idx, sample in enumerate(tqdm(test_data, desc="Generating images")):
        try:
            print(f"\n{'='*80}")
            print(f"[DEBUG] Processing sample {idx + start_idx}/{total_samples} (index {idx})")
            print(f"{'='*80}")
            
            user_id = sample['user_id']
            user_interactions = sample['interaction_sequence']
            print(f"[DEBUG] User ID: {user_id}, Number of interactions: {len(user_interactions)}")

            print(f"[DEBUG] Step 1: Selecting user history images...")
            history_data = select_user_history_images(user_interactions, image_dir, min_score=4.0, max_images=5)
            if len(history_data) == 0:
                print(f"[WARNING] No suitable history images for user {user_id}")
                failed_samples.append({'index': idx, 'reason': 'No suitable history images'})
                continue

            print(f"[DEBUG] Step 2: Loading {len(history_data)} history images...")
            history_image_paths = [path for path, _, _ in history_data]
            history_images = load_images(history_image_paths)

            if len(history_images) == 0:
                print(f"[WARNING] Could not load any history images for user {user_id}")
                failed_samples.append({'index': idx, 'reason': 'Could not load history images'})
                continue

            print(f"[DEBUG] Step 3: Selecting target for generation...")
            target_data = select_target_for_generation(user_interactions, captions, image_dir, seed=seed + idx)
            if target_data is None:
                print(f"[WARNING] Could not select target for user {user_id}")
                failed_samples.append({'index': idx, 'reason': 'Could not select target'})
                continue

            target_image_path, target_item_id, target_caption = target_data
            print(f"[DEBUG]   Target selected: item_id={target_item_id}, path={target_image_path}")

            if len(history_images) > 1:
                print(f"[DEBUG] Step 4: Checking IP-Adapter instances (have {len(history_images)} history images)...")
                num_loaded = 1
                if hasattr(pipe.unet, 'encoder_hid_proj') and hasattr(pipe.unet.encoder_hid_proj, 'image_projection_layers'):
                    num_loaded = len(pipe.unet.encoder_hid_proj.image_projection_layers)

                if num_loaded < len(history_images):
                    print(f"[DEBUG]   Loading additional IP-Adapter instances ({num_loaded} -> {len(history_images)})...")
                    load_multiple_ip_adapters(pipe, ip_adapter_path, ip_adapter_weight, len(history_images))
                else:
                    print(f"[DEBUG]   Sufficient IP-Adapter instances already loaded ({num_loaded})")

            print(f"[DEBUG] Step 5: Generating personalized image...")
            print(f"[DEBUG]   Prompt: {target_caption[:100]}..." if len(target_caption) > 100 else f"[DEBUG]   Prompt: {target_caption}")
            print(f"[DEBUG]   Inference steps: {num_inference_steps}, Guidance scale: {guidance_scale}, IP-Adapter scale: {ip_adapter_scale}")
            generated_image = generate_personalized_image(
                pipe=pipe,
                history_images=history_images,
                target_caption=target_caption,
                target_image_path=target_image_path,
                prompt=target_caption,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                ip_adapter_scale=ip_adapter_scale,
                seed=seed + idx
            )
            print(f"[DEBUG]   ✓ Image generated successfully, size: {generated_image.size}")

            print(f"[DEBUG] Step 6: Saving generated image...")
            safe_filename = f"{user_id}_{target_item_id}".replace('/', '_').replace('\\', '_')
            output_path = os.path.join(generated_dir, f"{safe_filename}_generated.png")
            generated_image.save(output_path)
            print(f"[DEBUG]   ✓ Image saved to: {output_path}")

            print(f"[DEBUG] Step 7: Calculating evaluation metrics...")
            metrics = evaluator.evaluate_generated_image(
                generated_image=generated_image,
                target_image_path=target_image_path,
                history_images=history_images,
                target_caption=target_caption,
                generated_image_path=output_path,
                user_id=user_id
            )

            print(f"[DEBUG] Step 8: Calculating verifier score...")
            verifier_score = verifier.score_image(user_id, output_path)
            metrics['verifier_score'] = verifier_score

            print(f"[DEBUG] Final metrics for sample {idx + start_idx}:")
            for key, value in metrics.items():
                if value is not None:
                    print(f"[DEBUG]   {key}: {value:.4f}")
                else:
                    print(f"[DEBUG]   {key}: None (not calculated)")

            result = {
                'index': idx + start_idx,
                'user_id': user_id,
                'target_item_id': target_item_id,
                'num_history_images': len(history_images),
                'target_caption': target_caption,
                'output_path': output_path,
                'target_image_path': target_image_path,
                'success': True,
                'metrics': metrics,
                'history_scores': [score for _, _, score in history_data]
            }
            results.append(result)

            for key in all_metrics.keys():
                if metrics.get(key) is not None:
                    all_metrics[key].append(metrics[key])
            
            print(f"[DEBUG] ✓ Sample {idx + start_idx} processed successfully")

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            failed_samples.append({'index': idx, 'reason': str(e)})
            continue

    avg_metrics = {}
    for key, values in all_metrics.items():
        if values:
            avg_metrics[f'avg_{key}'] = float(np.mean(values))
            avg_metrics[f'std_{key}'] = float(np.std(values))

    results_path = os.path.join(output_dir, "generation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_samples': len(test_data),
            'successful': len(results),
            'failed': len(failed_samples),
            'results': results,
            'failed_samples': failed_samples,
            'average_metrics': avg_metrics,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("FLICKR-AES Generation and Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {len(test_data)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed_samples)}")
    print("\nAverage Metrics:")
    for key in all_metrics.keys():
        metric_name = key.upper()
        avg_key = f'avg_{key}'
        std_key = f'std_{key}'
        if avg_key in avg_metrics:
            value = avg_metrics[avg_key]
            std_value = avg_metrics.get(std_key, 0)
            print(f"  {metric_name:25s}: {value:.4f} ± {std_value:.4f}")
        else:
            print(f"  {metric_name:25s}: N/A (all values were None)")
    print(f"\nResults saved to {results_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="FLICKR-AES dataset personalized image generation script")
    parser.add_argument(
        "--test_json",
        type=str,
        default="{FLICKR_AES_BASE_PATH}/processed_dataset/test.json",
        help="Test JSON path"
    )
    parser.add_argument(
        "--captions_path",
        type=str,
        default="{FLICKR_AES_BASE_PATH}/FLICKR_captions.json",
        help="Image captions JSON path"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="{FLICKR_AES_BASE_PATH}/40K",
        help="Image directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="{FLICKR_AES_BASE_PATH}/ip_adapter_generated",
        help="Output directory"
    )
    parser.add_argument(
        "--sd_model_path",
        type=str,
        default="{SD15_MODEL_PATH}",
        help="Stable Diffusion model path"
    )
    parser.add_argument(
        "--ip_adapter_path",
        type=str,
        default="{IP_ADAPTER_PATH}",
        help="IP-Adapter model path"
    )
    parser.add_argument(
        "--ip_adapter_weight",
        type=str,
        default="ip-adapter_sd15.bin",
        help="IP-Adapter weight filename"
    )
    parser.add_argument(
        "--verifier_model_path",
        type=str,
        default="{FLICKR_AES_BASE_PATH}/verifier_checkpoints/best_model.pth",
        help="Trained verifier model path"
    )
    parser.add_argument(
        "--verifier_user_map_path",
        type=str,
        default="{FLICKR_AES_BASE_PATH}/verifier_checkpoints/user_map.json",
        help="Verifier user map file path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None for all)"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale"
    )
    parser.add_argument(
        "--ip_adapter_scale",
        type=float,
        default=0.6,
        help="IP-Adapter scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--styles_path",
        type=str,
        default="{FLICKR_AES_BASE_PATH}/FLICKR_styles.json",
        help="User style preferences JSON path"
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    process_flickr_dataset(
        test_json_path=args.test_json,
        captions_path=args.captions_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        sd_model_path=args.sd_model_path,
        ip_adapter_path=args.ip_adapter_path,
        ip_adapter_weight=args.ip_adapter_weight,
        verifier_model_path=args.verifier_model_path,
        verifier_user_map_path=args.verifier_user_map_path,
        device=args.device,
        max_samples=args.max_samples,
        start_idx=args.start_idx,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        ip_adapter_scale=args.ip_adapter_scale,
        seed=args.seed,
        styles_path=args.styles_path
    )


if __name__ == "__main__":
    main()
