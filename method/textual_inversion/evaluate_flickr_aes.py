"""
FLICKR-AES dataset evaluation script

This script evaluates the performance of Textual Inversion trained embeddings on the FLICKR-AES test set.

Evaluation metrics include:
1. LPIPS (Learned Perceptual Image Patch Similarity)
2. SSIM (Structural Similarity Index Measure)
3. CPS (CLIP Personalized Score)
4. CPIS (CLIP Personalized Image Score)
5. HPSv2 - Human Preference Score
6. LAION Aesthetic Score
7. Verifier Score

Usage:
    python evaluate_flickr_aes.py --test_json <test_json_path> --embedding_path <embedding_path> [other args]

"""

import os
import sys
import json
import time

import torch
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Optional
import argparse
from dataclasses import dataclass

# Metrics
import lpips
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import clip

# HPSv2 and LAION aesthetic
HPSv2_AVAILABLE = False
hpsv2_module = None
try:
    import hpsv2
    hpsv2_module = hpsv2
    HPSv2_AVAILABLE = True
except ImportError:
    try:
        from hpsv2 import HPSv2
        # If HPSv2 is a class, wrap it
        class HPSv2Wrapper:
            def __init__(self, *args, **kwargs):
                self.model = HPSv2(*args, **kwargs)
            def score(self, imgs_path, prompt, hps_version="v2.1"):
                return self.model.score(imgs_path, prompt, hps_version=hps_version)
        hpsv2_module = type('hpsv2_module', (), {'score': lambda imgs_path, prompt, hps_version="v2.1": HPSv2().score(imgs_path, prompt, hps_version)})
        HPSv2_AVAILABLE = True
    except ImportError:
        print("Warning: hpsv2 not installed. Install with: pip install hpsv2")

try:
    from transformers import AutoProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
    print("transformers imported successfully for LAION aesthetic")
except ImportError as e:
    print(f"Warning: transformers not installed for LAION aesthetic: {e}")
    TRANSFORMERS_AVAILABLE = False
    AutoProcessor = None
    AutoModel = None
except Exception as e:
    print(f"Warning: Failed to import transformers (may cause issues): {e}")
    TRANSFORMERS_AVAILABLE = False
    AutoProcessor = None
    AutoModel = None

StableDiffusionPipeline = None

from torchvision import transforms


@dataclass
class EvalConfig:
    test_json: str = "{FLICKR_AES_BASE_PATH}/processed_dataset/test.json"
    embedding_path: str = "{FLICKR_AES_BASE_PATH}/textual_inversion_sd15/learned_embeds.bin"
    sd15_path: str = "{SD15_MODEL_PATH}"
    target_token: str = "[V]"
    output_dir: str = "{FLICKR_AES_BASE_PATH}/evaluation_results"
    generated_images_dir: str = "{FLICKR_AES_BASE_PATH}/evaluation_generated_images"
    images_dir: str = "{FLICKR_AES_BASE_PATH}/40K"
    captions_json: str = "{FLICKR_AES_BASE_PATH}/FLICKR_captions_masked.json"
    verifier_model_path: str = "{FLICKR_AES_BASE_PATH}/verifier_checkpoints/best_model.pth"
    verifier_user_map_path: str = "{FLICKR_AES_BASE_PATH}/verifier_checkpoints/user_map.json"
    styles_path: str = "{FLICKR_AES_BASE_PATH}/FLICKR_styles.json"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    image_size: int = 512
    batch_size: int = 4
    num_samples: int = None  # None means evaluate all samples
    seed: int = 42


class VerifierScorer:

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

        import sys
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
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
        
        print(f"[{time.strftime('%H:%M:%S')}] Loading CLIP model for feature extraction (expected dim: {input_dim})...")
        try:
            if input_dim == 1024:
                if TRANSFORMERS_AVAILABLE:
                    from transformers import CLIPProcessor, CLIPModel
                    local_model_path = "{CLIP_VIT_H14_MODEL_PATH}"
                    
                    if os.path.exists(local_model_path):
                        try:
                            self.clip_processor = CLIPProcessor.from_pretrained(local_model_path, local_files_only=True)
                            self.clip_model = CLIPModel.from_pretrained(local_model_path, local_files_only=True).to(device)
                            self.clip_model.eval()
                            print(f"[{time.strftime('%H:%M:%S')}] ✓ CLIP ViT-H-14 loaded successfully from {local_model_path}")
                        except Exception as e:
                            error_msg = f"Failed to load CLIP ViT-H-14 from {local_model_path}: {e}"
                            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
                            import traceback
                            traceback.print_exc()
                            raise RuntimeError(error_msg)
                    else:
                        error_msg = f"CLIP ViT-H-14 model not found at {local_model_path}"
                        print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
                        raise RuntimeError(error_msg)
                else:
                    error_msg = "Transformers not available, cannot load CLIP ViT-H-14"
                    print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
                    raise RuntimeError(error_msg)
            else:
                self._load_alternative_clip(input_dim)
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            error_msg = f"Failed to load CLIP model: {e}"
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)

        print(f"[{time.strftime('%H:%M:%S')}] Verifier model loaded from {model_path}")

    def _load_alternative_clip(self, expected_dim: int):
        try:
            if expected_dim == 768:
                clip_model_name = "ViT-L/14"
            elif expected_dim == 512:
                clip_model_name = "ViT-B/32"
            else:
                clip_model_name = "ViT-L/14"
                print(f"[{time.strftime('%H:%M:%S')}] Warning: Expected dim {expected_dim} not standard, using ViT-L/14 (768 dim)")
            
            print(f"[{time.strftime('%H:%M:%S')}] Loading CLIP {clip_model_name}...")
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
            self.clip_model.eval()
            self.clip_transform = transforms.Compose([
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                   std=(0.26862954, 0.26130258, 0.27577711))
            ])
            print(f"[{time.strftime('%H:%M:%S')}] ✓ CLIP {clip_model_name} loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load alternative CLIP model ({clip_model_name}): {e}"
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)

    def score_image(self, user_id: str, image_path: str) -> Optional[float]:
        try:
            if user_id not in self.user_map:
                return None

            user_idx = self.user_map[user_id]

            if self.clip_model is None:
                return None

            if not os.path.exists(image_path):
                return None

            image = Image.open(image_path).convert("RGB")

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
                    if image_features.shape[-1] > self.input_dim:
                        image_features = image_features[:, :self.input_dim]
                    else:
                        padding = torch.zeros(
                            image_features.shape[0], 
                            self.input_dim - image_features.shape[-1],
                            device=image_features.device
                        )
                        image_features = torch.cat([image_features, padding], dim=-1)

            with torch.no_grad():
                user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                score = self.model(user_tensor, image_features)
                score_value = score.item()
                return score_value

        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error scoring image with verifier: {e}")
            return None


class FLICKRAESEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        
        print(f"[{time.strftime('%H:%M:%S')}] Initializing FLICKRAESEvaluator...")
        
        # Set random seed for reproducibility
        print(f"[{time.strftime('%H:%M:%S')}] Setting random seed to {config.seed}...")
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        
        print(f"[{time.strftime('%H:%M:%S')}] Checking CUDA...")
        if torch.cuda.is_available():
            print(f"[{time.strftime('%H:%M:%S')}] CUDA available: {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] CUDA not available, using CPU")
            self.device = torch.device("cpu")
        
        
        print(f"[{time.strftime('%H:%M:%S')}] Creating output directories...")
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.generated_images_dir, exist_ok=True)
        
        
        print(f"[{time.strftime('%H:%M:%S')}] Starting metrics initialization...")
        self._init_metrics()
        
        print(f"[{time.strftime('%H:%M:%S')}] Starting model loading...")
        self._load_model()
        
        print(f"[{time.strftime('%H:%M:%S')}] Loading test data...")
        self._load_test_data()
        
        print(f"[{time.strftime('%H:%M:%S')}] Loading captions...")
        self._load_captions()
        
        print(f"[{time.strftime('%H:%M:%S')}] Loading user styles...")
        self._load_user_styles()
        
        print(f"[{time.strftime('%H:%M:%S')}] Loading verifier scorer...")
        self._load_verifier()
        
        print(f"[{time.strftime('%H:%M:%S')}] FLICKRAESEvaluator initialized successfully")
    
    def _init_metrics(self):
        """Initialize all evaluation metrics"""
        print(f"[{time.strftime('%H:%M:%S')}] Initializing metrics...")
        
        # LPIPS
        print(f"[{time.strftime('%H:%M:%S')}] Loading LPIPS...", flush=True)
        try:
            self.lpips_metric = lpips.LPIPS(net='vgg').to(self.device)
            self.lpips_metric.eval()
            print(f"[{time.strftime('%H:%M:%S')}] LPIPS loaded", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ERROR loading LPIPS: {e}", flush=True)
            raise
        
        # SSIM
        print(f"[{time.strftime('%H:%M:%S')}] Loading SSIM...")
        self.ssim_metric = SSIM(data_range=2.0).to(self.device)
        print(f"[{time.strftime('%H:%M:%S')}] SSIM loaded")
        
        # CLIP for CPS and CPIS
        print(f"[{time.strftime('%H:%M:%S')}] Loading CLIP...", flush=True)
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            print(f"[{time.strftime('%H:%M:%S')}] CLIP loaded", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ERROR loading CLIP: {e}", flush=True)
            raise
        
        # CLIP preprocessing for images in [0, 1] range
        self.clip_transform = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                               std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        # HPSv2
        self.hpsv2_model = None
        self.hpsv2_failed = False
        self.hpsv2_error_reason = None
        if HPSv2_AVAILABLE and hpsv2_module is not None:
            try:
                if hasattr(hpsv2_module, 'score'):
                    self.hpsv2_model = hpsv2_module
                    print(f"[{time.strftime('%H:%M:%S')}] HPSv2 initialized (module mode)")
                else:
                    try:
                        self.hpsv2_model = hpsv2_module.HPSv2(device=self.device)
                        print(f"[{time.strftime('%H:%M:%S')}] HPSv2 initialized (class mode)")
                    except (TypeError, AttributeError):
                        try:
                            self.hpsv2_model = hpsv2_module.HPSv2()
                            if hasattr(self.hpsv2_model, 'to'):
                                self.hpsv2_model = self.hpsv2_model.to(self.device)
                            print(f"[{time.strftime('%H:%M:%S')}] HPSv2 initialized (class mode, no device)")
                        except Exception as e2:
                            error_msg = f"HPSv2 initialization failed: {e2}"
                            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
                            import traceback
                            traceback.print_exc()
                            raise RuntimeError(error_msg)
            except Exception as e:
                error_msg = f"Failed to initialize HPSv2: {e}"
                print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(error_msg)
        else:
            error_msg = "HPSv2 is not available. Please install hpsv2: pip install hpsv2"
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
            raise RuntimeError(error_msg)
        
        print(f"[{time.strftime('%H:%M:%S')}] Skipping LAION aesthetic predictor (will load on-demand if needed)")
        self.laion_aesthetic_model = None
        self._laion_loaded = False
        
        print(f"[{time.strftime('%H:%M:%S')}] Metrics initialized")
    
    def _load_model(self):
        """Load Stable Diffusion model and embeddings"""
        print(f"[{time.strftime('%H:%M:%S')}] Loading Stable Diffusion model...", flush=True)
        
        global StableDiffusionPipeline
        if StableDiffusionPipeline is None:
            print(f"[{time.strftime('%H:%M:%S')}] Importing diffusers (delayed load)...", flush=True)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            import sys
            try:
                print(f"[{time.strftime('%H:%M:%S')}] Step 1: Importing diffusers base module...", flush=True)
                import diffusers
                print(f"[{time.strftime('%H:%M:%S')}] Step 2: Getting StableDiffusionPipeline class...", flush=True)
                import importlib
                pipeline_module = importlib.import_module('diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion')
                StableDiffusionPipeline = pipeline_module.StableDiffusionPipeline
                print(f"[{time.strftime('%H:%M:%S')}] diffusers imported successfully", flush=True)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] ERROR: Failed to import diffusers: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to import diffusers: {e}")
        
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Creating pipeline from {self.config.sd15_path}...", flush=True)
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.config.sd15_path, 
                safety_checker=None,
                torch_dtype=torch.float16
            )
            print(f"[{time.strftime('%H:%M:%S')}] Pipeline created, moving to device...", flush=True)
            self.pipe.to(self.device)
            print(f"[{time.strftime('%H:%M:%S')}] Stable Diffusion model loaded", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ERROR loading Stable Diffusion model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        # Load embeddings
        print(f"[{time.strftime('%H:%M:%S')}] Loading embeddings from {self.config.embedding_path}...")
        embeddings = torch.load(self.config.embedding_path, map_location=self.device)
        
        # Get the embedding tensor (should be shape [8, 768])
        if self.config.target_token in embeddings:
            embedding_tensor = embeddings[self.config.target_token]
        else:
            # If key doesn't match, take the first value
            embedding_tensor = list(embeddings.values())[0]
        
        print(f"[{time.strftime('%H:%M:%S')}] Embedding shape: {embedding_tensor.shape}")
        
        # Get placeholder tokens (should be <v_0> to <v_7>)
        tokenizer = self.pipe.tokenizer
        placeholder_tokens = [f"<v_{i}>" for i in range(embedding_tensor.shape[0])]
        
        # Check if tokens exist, if not add them
        existing_tokens = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        if any(tid == tokenizer.unk_token_id for tid in existing_tokens):
            # Need to add tokens
            num_added = tokenizer.add_tokens(placeholder_tokens)
            if num_added != embedding_tensor.shape[0]:
                raise ValueError(f"Failed to add tokens. Expected {embedding_tensor.shape[0]}, got {num_added}")
            self.pipe.text_encoder.resize_token_embeddings(len(tokenizer))
        
        # Set embeddings
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        with torch.no_grad():
            self.pipe.text_encoder.get_input_embeddings().weight[placeholder_token_ids] = embedding_tensor.to(
                self.pipe.text_encoder.get_input_embeddings().weight.dtype
            )
        
        print(f"[{time.strftime('%H:%M:%S')}] Model and embeddings loaded")
    
    def _load_test_data(self):
        """Load test dataset"""
        print(f"[{time.strftime('%H:%M:%S')}] Loading test data from {self.config.test_json}...")
        with open(self.config.test_json, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        if self.config.num_samples is not None:
            self.test_data = self.test_data[:self.config.num_samples]
        
        print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(self.test_data)} test samples")
    
    def _load_captions(self):
        """Load captions from JSON file"""
        print(f"[{time.strftime('%H:%M:%S')}] Loading captions...")
        try:
            with open(self.config.captions_json, 'r', encoding='utf-8') as f:
                self.captions = json.load(f)
            print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(self.captions)} captions")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Warning: Failed to load captions: {e}")
            self.captions = {}
    
    def _load_user_styles(self):
        """Load user styles from JSON file"""
        if not os.path.exists(self.config.styles_path):
            print(f"[{time.strftime('%H:%M:%S')}] Warning: User styles file not found at {self.config.styles_path}")
            self.user_styles = {}
            return
        
        try:
            with open(self.config.styles_path, 'r', encoding='utf-8') as f:
                styles_data = json.load(f)
            self.user_styles = {}
            for item in styles_data:
                worker_id = item.get('worker', '')
                style_text = item.get('style', '')
                if worker_id and style_text:
                    self.user_styles[worker_id] = style_text
            print(f"[{time.strftime('%H:%M:%S')}] Loaded styles for {len(self.user_styles)} users")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Warning: Failed to load user styles: {e}")
            self.user_styles = {}
    
    def _load_verifier(self):
        """Load verifier scorer"""
        try:
            self.verifier = VerifierScorer(
                self.config.verifier_model_path,
                self.config.verifier_user_map_path,
                device=str(self.device)
            )
            print(f"[{time.strftime('%H:%M:%S')}] Verifier scorer loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load verifier scorer: {e}"
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)
    
    def _get_caption(self, item_id: str) -> Optional[str]:
        """Get caption for an item_id"""
        # Remove .jpg extension if present
        item_id_key = item_id.replace('.jpg', '').replace('.png', '')
        return self.captions.get(item_id_key)
    
    def _get_user_preference_text(self, user_id: str) -> Optional[str]:
        """Get user preference text from user_id"""
        if user_id not in self.user_styles:
            return None
        
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
        # Deduplicate and combine
        unique_keywords = list(dict.fromkeys(keywords))  # Deduplicate while preserving order
        preference_text = ", ".join(unique_keywords[:20])  # Limit length
        return preference_text if preference_text else None
    
    def _image_to_tensor(self, image_path: str) -> torch.Tensor:
        """Load image and convert to tensor in [0, 1] range"""
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.config.image_size, self.config.image_size), Image.BICUBIC)
        
        transform = transforms.Compose([
            transforms.ToTensor()  # [0, 1]
        ])
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def _calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate LPIPS between two images"""
        img1_norm = (img1 * 2.0) - 1.0  # [0,1] -> [-1,1]
        img2_norm = (img2 * 2.0) - 1.0  # [0,1] -> [-1,1]
        with torch.no_grad():
            score = self.lpips_metric(img1_norm, img2_norm)
        return score.item()
    
    def _calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate SSIM between two images"""
        with torch.no_grad():
            score = self.ssim_metric(img1, img2)
        score_value = score.item()
        
        if score_value < 0:
            print(f"[WARNING] SSIM calculated negative value: {score_value:.6f}")
        
        return score_value
    
    def _calculate_clip_similarity(self, image: torch.Tensor, text: str) -> float:
        """Calculate CLIP similarity between image and text"""
        with torch.no_grad():
            image_clip = self.clip_transform(image)
            image_features = self.clip_model.encode_image(image_clip)
            text_tokens = clip.tokenize([text]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
        return similarity
    
    def _calculate_cps(self, image: torch.Tensor, user_id: str) -> float:
        """Calculate CPS: CLIP similarity between image and user style text"""
        preference_text = self._get_user_preference_text(user_id)
        if preference_text is None:
            return None
        similarity = self._calculate_clip_similarity(image, preference_text)
        return similarity
    
    def _calculate_clip_image_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate CLIP similarity between two images (CPIS)"""
        with torch.no_grad():
            img1_clip = self.clip_transform(img1)
            img2_clip = self.clip_transform(img2)
            features1 = self.clip_model.encode_image(img1_clip)
            features2 = self.clip_model.encode_image(img2_clip)
            features1 = features1 / features1.norm(dim=-1, keepdim=True)
            features2 = features2 / features2.norm(dim=-1, keepdim=True)
            similarity = (features1 @ features2.T).item()
        return similarity
    
    def _calculate_hpsv2(self, image_path: str, prompt: str) -> float:
        """Calculate HPSv2 score"""
        if self.hpsv2_model is None or self.hpsv2_failed:
            if self.hpsv2_error_reason:
                error_msg = f"HPSv2 calculation failed: {self.hpsv2_error_reason}"
                print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            else:
                error_msg = "HPSv2 model is not initialized"
                print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
                raise RuntimeError(error_msg)
        
        try:
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
                return float(result[0])
            elif isinstance(result, torch.Tensor):
                return float(result.item())
            elif isinstance(result, np.ndarray):
                return float(result.item() if result.size == 1 else result[0])
            else:
                return float(result)
        except FileNotFoundError as e:
            error_msg = f"HPSv2 calculation failed due to missing file: {e}. Please reinstall hpsv2 or download the missing file."
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)
        except Exception as e:
            # Check if it's a network-related error (Hugging Face Hub download failure)
            error_str = str(e)
            if "Hub" in error_str or "local cache" in error_str or "Internet connection" in error_str or "LocalEntryNotFoundError" in error_str:
                error_msg = f"HPSv2 calculation failed: Model files not found in cache. HPSv2 requires model files to be pre-downloaded. Error: {type(e).__name__}: {error_str[:200]}"
                print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
                print(f"[{time.strftime('%H:%M:%S')}] Note: HPSv2 models need to be pre-downloaded when offline mode is enabled")
                import traceback
                traceback.print_exc()
                raise RuntimeError(error_msg)
            # Other errors
            error_msg = f"HPSv2 calculation failed: {type(e).__name__}: {str(e)[:200]}"
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)
    
    def _calculate_laion_aesthetic(self, image: Image.Image) -> float:
        """Calculate LAION aesthetic score using CLIP + Linear Regression Head"""
        if not self._laion_loaded:
            print(f"[{time.strftime('%H:%M:%S')}] Loading LAION aesthetic predictor (linear head)...")
            
            model_path = "{LAION_AESTHETIC_MODEL_PATH}"
            
            if not os.path.exists(model_path):
                error_msg = f"ERROR: LAION aesthetic model not found at {model_path}"
                print(f"[{time.strftime('%H:%M:%S')}] {error_msg}")
                if not hasattr(self, '_laion_error_logged'):
                    self.laion_error_reason = error_msg
                    self._laion_error_logged = True
                return None
            
            try:
                import torch.nn as nn
                self.laion_aesthetic_model = nn.Linear(512, 1)
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                self.laion_aesthetic_model.load_state_dict(state_dict)
                self.laion_aesthetic_model.eval()
                self.laion_aesthetic_model = self.laion_aesthetic_model.to(self.device).float()
                self._laion_loaded = True
                print(f"[{time.strftime('%H:%M:%S')}] LAION aesthetic predictor loaded successfully")
            except Exception as e:
                error_msg = f"ERROR: Failed to load LAION aesthetic model: {type(e).__name__}: {str(e)}"
                print(f"[{time.strftime('%H:%M:%S')}] {error_msg}")
                if not hasattr(self, '_laion_error_logged'):
                    self.laion_error_reason = error_msg
                    self._laion_error_logged = True
                return None
        
        if self.laion_aesthetic_model is None:
            if hasattr(self, 'laion_error_reason') and not hasattr(self, '_laion_error_logged'):
                print(f"[{time.strftime('%H:%M:%S')}] LAION aesthetic calculation skipped: {self.laion_error_reason}")
                self._laion_error_logged = True
            return None
        
        try:
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.float()
                aesthetic_score = self.laion_aesthetic_model(image_features)
                score = aesthetic_score.item()
            return float(score)
        except Exception as e:
            error_msg = f"LAION aesthetic calculation failed: {type(e).__name__}: {str(e)}"
            if not hasattr(self, '_laion_calc_error_logged'):
                print(f"[{time.strftime('%H:%M:%S')}] Error: {error_msg}")
                self._laion_calc_error_logged = True
                if not hasattr(self, 'laion_error_reason'):
                    self.laion_error_reason = error_msg
            return None
    
    def _replace_token_in_prompt(self, prompt: str) -> str:
        """Replace [V] token with placeholder tokens"""
        placeholder_tokens_str = " ".join([f"<v_{i}>" for i in range(8)])
        return prompt.replace(self.config.target_token, placeholder_tokens_str)
    
    def _select_target_and_history(self, sample: Dict) -> tuple:
        """Select target image and history images from interaction sequence"""
        user_interactions = sample.get('interaction_sequence', [])
        
        # Select target: choose a high-score image (>= 4.0) that we can generate
        target_candidate = None
        for interaction in user_interactions:
            if interaction.get('score', 0) >= 4.0:
                item_id = interaction.get('item_id', '')
                image_path = interaction.get('image_path')
                if item_id and image_path and os.path.exists(image_path):
                    target_candidate = (item_id, image_path)
                    break
        
        if target_candidate is None:
            # Fallback: use first available image
            for interaction in user_interactions:
                item_id = interaction.get('item_id', '')
                image_path = interaction.get('image_path')
                if item_id and image_path and os.path.exists(image_path):
                    target_candidate = (item_id, image_path)
                    break
        
        if target_candidate is None:
            return None, []
        
        target_item_id, target_image_path = target_candidate
        
        # Select history images: high-score images (>= 4.0), up to 10
        history_items = []
        for interaction in user_interactions:
            if interaction.get('score', 0) >= 4.0:
                item_id = interaction.get('item_id', '')
                image_path = interaction.get('image_path')
                if item_id and image_path and os.path.exists(image_path) and item_id != target_item_id:
                    history_items.append({
                        'item_id': item_id,
                        'image_path': image_path,
                        'score': interaction.get('score', 0)
                    })
                    if len(history_items) >= 10:
                        break
        
        return (target_item_id, target_image_path), history_items
    
    def evaluate(self):
        """Run evaluation on all test samples"""
        start_time = time.time()
        print(f"\n[{time.strftime('%H:%M:%S')}] Starting evaluation on {len(self.test_data)} samples...")
        print(f"[{time.strftime('%H:%M:%S')}] Estimated time: {len(self.test_data) * 40 / 60:.1f} - {len(self.test_data) * 70 / 60:.1f} minutes")
        
        results = []
        all_metrics = {
            'lpips_target': [],
            'lpips_history_avg': [],
            'ssim_target': [],
            'ssim_history_avg': [],
            'cps': [],
            'cpis_history_avg': [],
            'hpsv2': [],
            'laion_aesthetic': [],
            'verifier_score': []
        }
        
        # Batch generation: prepare sample data
        batch_samples = []
        for idx, sample in enumerate(self.test_data):
            try:
                user_id = sample.get('user_id')
                if not user_id:
                    continue
                
                # Select target and history images
                target_data, history_items = self._select_target_and_history(sample)
                if target_data is None:
                    continue
                
                target_item_id, target_image_path = target_data
                
                # Get caption
                target_caption = self._get_caption(target_item_id)
                if not target_caption:
                    continue
                
                # Replace [V] with placeholder tokens
                generation_prompt = self._replace_token_in_prompt(target_caption)
                
                save_path = os.path.join(
                    self.config.generated_images_dir,
                    f"{user_id}_{target_item_id.replace('/', '_').replace('.jpg', '').replace('.png', '')}.png"
                )
                
                batch_samples.append({
                    'idx': idx,
                    'sample': sample,
                    'user_id': user_id,
                    'target_item_id': target_item_id,
                    'target_image_path': target_image_path,
                    'target_caption': target_caption,
                    'generation_prompt': generation_prompt,
                    'save_path': save_path,
                    'history_items': history_items
                })
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Warning: Failed to prepare sample {idx}: {e}")
                continue
        
        print(f"[{time.strftime('%H:%M:%S')}] Prepared {len(batch_samples)} samples for evaluation")
        print(f"[{time.strftime('%H:%M:%S')}] Using batch size: {self.config.batch_size}")
        
        # Batch generate images
        for batch_start in tqdm(range(0, len(batch_samples), self.config.batch_size), desc="Generating batches"):
            batch_end = min(batch_start + self.config.batch_size, len(batch_samples))
            current_batch = batch_samples[batch_start:batch_end]
            
            # Prepare batch prompts and generators
            batch_prompts = [item['generation_prompt'] for item in current_batch]
            batch_generators = [
                torch.Generator(device=self.device).manual_seed(self.config.seed + item['idx'])
                for item in current_batch
            ]
            
            # Batch generate images
            try:
                with torch.no_grad():
                    if len(batch_prompts) == 1:
                        # Single sample, generate directly
                        generated_images = [self.pipe(
                            batch_prompts[0],
                            num_inference_steps=self.config.num_inference_steps,
                            guidance_scale=self.config.guidance_scale,
                            generator=batch_generators[0]
                        ).images[0]]
                    else:
                        # Multiple samples, generate one by one (cannot truly batch due to different prompts)
                        # But still get speedup through batch preparation and GPU optimization
                        generated_images = []
                        for i, (prompt, gen) in enumerate(zip(batch_prompts, batch_generators)):
                            img = self.pipe(
                                prompt,
                                num_inference_steps=self.config.num_inference_steps,
                                guidance_scale=self.config.guidance_scale,
                                generator=gen
                            ).images[0]
                            generated_images.append(img)
                            # Clear cache after each image generation (avoid memory overflow)
                            if (i + 1) % 2 == 0 and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    
                    # Save generated images
                    for item, img in zip(current_batch, generated_images):
                        img.save(item['save_path'])
                        
            except Exception as gen_e:
                print(f"[{time.strftime('%H:%M:%S')}] ERROR: Failed to generate batch {batch_start}-{batch_end}: {gen_e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process metrics calculation for each sample
            for item, generated_image in zip(current_batch, generated_images):
                idx = item['idx']
                user_id = item['user_id']
                target_item_id = item['target_item_id']
                target_image_path = item['target_image_path']
                target_caption = item['target_caption']
                generation_prompt = item['generation_prompt']
                save_path = item['save_path']
                history_items = item['history_items']
                
                try:
                    # Load target image as tensor
                    target_tensor = self._image_to_tensor(target_image_path)
                    generated_tensor = self._image_to_tensor(save_path)
                    
                    lpips_target = self._calculate_lpips(generated_tensor, target_tensor)
                    ssim_target = self._calculate_ssim(generated_tensor, target_tensor)
                    
                    cps_score = self._calculate_cps(generated_tensor, user_id)
                    
                    # Calculate metrics with history images
                    lpips_history_scores = []
                    ssim_history_scores = []
                    cpis_history_scores = []
                    
                    for hist_item in history_items:
                        hist_image_path = hist_item.get('image_path')
                        if hist_image_path and os.path.exists(hist_image_path):
                            try:
                                hist_tensor = self._image_to_tensor(hist_image_path)
                                lpips_hist = self._calculate_lpips(generated_tensor, hist_tensor)
                                lpips_history_scores.append(lpips_hist)
                                ssim_hist = self._calculate_ssim(generated_tensor, hist_tensor)
                                ssim_history_scores.append(ssim_hist)
                                cpis_hist = self._calculate_clip_image_similarity(generated_tensor, hist_tensor)
                                cpis_history_scores.append(cpis_hist)
                            except Exception as e:
                                print(f"[{time.strftime('%H:%M:%S')}] Warning: Failed to process history image {hist_image_path}: {e}")
                                continue
                    
                    # Calculate averages
                    lpips_history_avg = np.mean(lpips_history_scores) if lpips_history_scores else None
                    ssim_history_avg = np.mean(ssim_history_scores) if ssim_history_scores else None
                    cpis_history_avg = np.mean(cpis_history_scores) if cpis_history_scores else None
                    
                    # Other metrics
                    hpsv2_score = self._calculate_hpsv2(save_path, target_caption)
                    laion_score = self._calculate_laion_aesthetic(generated_image)
                    
                    # Verifier score
                    verifier_score = None
                    if self.verifier is not None:
                        verifier_score = self.verifier.score_image(user_id, save_path)
                    
                    # Store results
                    result = {
                        'sample_idx': idx,
                        'user_id': user_id,
                        'target_item_id': target_item_id,
                        'prompt': target_caption,
                        'generation_prompt': generation_prompt,
                        'generated_image_path': save_path,
                        'target_image_path': target_image_path,
                        'num_history_images': len(history_items),
                        'lpips_target': lpips_target,
                        'lpips_history_avg': lpips_history_avg,
                        'ssim_target': ssim_target,
                        'ssim_history_avg': ssim_history_avg,
                        'cps': cps_score,
                        'cpis_history_avg': cpis_history_avg,
                        'hpsv2': hpsv2_score,
                        'laion_aesthetic': laion_score,
                        'verifier_score': verifier_score
                    }
                    results.append(result)
                    
                    # Accumulate metrics
                    all_metrics['lpips_target'].append(lpips_target)
                    if lpips_history_avg is not None:
                        all_metrics['lpips_history_avg'].append(lpips_history_avg)
                    all_metrics['ssim_target'].append(ssim_target)
                    if ssim_history_avg is not None:
                        all_metrics['ssim_history_avg'].append(ssim_history_avg)
                    if cps_score is not None:
                        all_metrics['cps'].append(cps_score)
                    if cpis_history_avg is not None:
                        all_metrics['cpis_history_avg'].append(cpis_history_avg)
                    if hpsv2_score is not None:
                        all_metrics['hpsv2'].append(hpsv2_score)
                    if laion_score is not None:
                        all_metrics['laion_aesthetic'].append(laion_score)
                    if verifier_score is not None:
                        all_metrics['verifier_score'].append(verifier_score)
                    
                    if (idx + 1) % 10 == 0:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / (idx + 1)
                        remaining = avg_time * (len(self.test_data) - idx - 1)
                        print(f"[{time.strftime('%H:%M:%S')}] Progress: {idx+1}/{len(self.test_data)} samples, "
                              f"avg {avg_time:.1f}s/sample, ~{remaining/60:.1f} min remaining")
                    
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] Error evaluating sample {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                avg_metrics[f'avg_{metric_name}'] = np.mean(values)
                avg_metrics[f'std_{metric_name}'] = np.std(values)
            else:
                avg_metrics[f'avg_{metric_name}'] = None
                avg_metrics[f'std_{metric_name}'] = None
        
        # Print summary
        print("\n" + "="*80)
        print("Evaluation Summary")
        print("="*80)
        metric_display = [
            ('lpips_target', 'LPIPS (vs Target)'),
            ('lpips_history_avg', 'LPIPS (vs History Avg)'),
            ('ssim_target', 'SSIM (vs Target)'),
            ('ssim_history_avg', 'SSIM (vs History Avg)'),
            ('cps', 'CPS'),
            ('cpis_history_avg', 'CPIS (vs History Avg)'),
            ('hpsv2', 'HPSv2'),
            ('laion_aesthetic', 'LAION Aesthetic'),
            ('verifier_score', 'Verifier Score')
        ]
        
        for metric_name, display_name in metric_display:
            avg_key = f'avg_{metric_name}'
            std_key = f'std_{metric_name}'
            if avg_metrics[avg_key] is not None:
                print(f"{display_name:25s}: {avg_metrics[avg_key]:.4f} ± {avg_metrics[std_key]:.4f}")
            else:
                error_info = "N/A"
                if metric_name == 'hpsv2' and hasattr(self, 'hpsv2_error_reason'):
                    error_info = f"N/A ({self.hpsv2_error_reason})"
                elif metric_name == 'laion_aesthetic':
                    if hasattr(self, 'laion_error_reason'):
                        error_info = f"N/A ({self.laion_error_reason})"
                elif metric_name == 'verifier_score':
                    error_info = "N/A (verifier not loaded or user not in map)"
                elif metric_name in ['lpips_history_avg', 'ssim_history_avg', 'cpis_history_avg']:
                    error_info = "N/A (no valid history images)"
                print(f"{display_name:25s}: {error_info}")
        print("="*80)
        
        # Save results
        results_path = os.path.join(self.config.output_dir, "evaluation_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'test_json': self.config.test_json,
                    'embedding_path': self.config.embedding_path,
                    'num_samples': len(self.test_data),
                    'num_inference_steps': self.config.num_inference_steps,
                    'guidance_scale': self.config.guidance_scale
                },
                'average_metrics': avg_metrics,
                'per_sample_results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {results_path}")
        
        return avg_metrics, results


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Starting main()...", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] Python version: {sys.version}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] PyTorch version: {torch.__version__}", flush=True)
    if torch.cuda.is_available():
        print(f"[{time.strftime('%H:%M:%S')}] CUDA available: {torch.cuda.get_device_name(0)}", flush=True)
    
    parser = argparse.ArgumentParser(description="Evaluate Textual Inversion on FLICKR-AES dataset")
    parser.add_argument("--test_json", type=str, 
                       default="{FLICKR_AES_BASE_PATH}/processed_dataset/test.json",
                       help="Path to test JSON file")
    parser.add_argument("--embedding_path", type=str,
                       default="{FLICKR_AES_BASE_PATH}/textual_inversion_sd15/learned_embeds.bin",
                       help="Path to learned embeddings")
    parser.add_argument("--sd15_path", type=str,
                       default="{SD15_MODEL_PATH}",
                       help="Path to Stable Diffusion 1.5 model")
    parser.add_argument("--output_dir", type=str,
                       default="{FLICKR_AES_BASE_PATH}/evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--generated_images_dir", type=str,
                       default="{FLICKR_AES_BASE_PATH}/evaluation_generated_images",
                       help="Directory to save generated images")
    parser.add_argument("--images_dir", type=str,
                       default="{FLICKR_AES_BASE_PATH}/40K",
                       help="Directory containing images")
    parser.add_argument("--captions_json", type=str,
                       default="{FLICKR_AES_BASE_PATH}/FLICKR_captions_masked.json",
                       help="Path to captions JSON file")
    parser.add_argument("--verifier_model_path", type=str,
                       default="{FLICKR_AES_BASE_PATH}/verifier_checkpoints/best_model.pth",
                       help="Path to verifier model")
    parser.add_argument("--verifier_user_map_path", type=str,
                       default="{FLICKR_AES_BASE_PATH}/verifier_checkpoints/user_map.json",
                       help="Path to verifier user map")
    parser.add_argument("--styles_path", type=str,
                       default="{FLICKR_AES_BASE_PATH}/FLICKR_styles.json",
                       help="Path to user styles JSON file")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (None for all)")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for parallel generation (default: 4, increase for faster GPU like L40)")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        test_json=args.test_json,
        embedding_path=args.embedding_path,
        sd15_path=args.sd15_path,
        output_dir=args.output_dir,
        generated_images_dir=args.generated_images_dir,
        images_dir=args.images_dir,
        captions_json=args.captions_json,
        verifier_model_path=args.verifier_model_path,
        verifier_user_map_path=args.verifier_user_map_path,
        styles_path=args.styles_path,
        num_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        batch_size=args.batch_size
    )
    
    try:
        print(f"[{time.strftime('%H:%M:%S')}] Creating FLICKRAESEvaluator...", flush=True)
        evaluator = FLICKRAESEvaluator(config)
        print(f"[{time.strftime('%H:%M:%S')}] Starting evaluation...", flush=True)
        evaluator.evaluate()
        print(f"[{time.strftime('%H:%M:%S')}] Evaluation completed", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

