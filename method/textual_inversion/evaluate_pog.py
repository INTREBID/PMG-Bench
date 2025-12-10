"""
POG dataset evaluation script

This script evaluates the performance of Textual Inversion trained embeddings on the POG test set.

Evaluation metrics include:
1. LPIPS (Learned Perceptual Image Patch Similarity)
2. SSIM (Structural Similarity Index Measure)
3. CPS (CLIP Personalized Score)
4. CPIS (CLIP Personalized Image Score)
5. HPSv2 - Human Preference Score
6. LAION Aesthetic Score

Usage:
    python evaluate_pog.py --test_json <test_json_path> --embedding_path <embedding_path> [other args]

"""

import os
import sys
import json
import time

import torch
import numpy as np
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
    test_json: str = "{POG_BASE_PATH}/processed_dataset/test.json"
    embedding_path: str = "{POG_BASE_PATH}/textual_inversion_sd15/learned_embeds.bin"
    sd15_path: str = "{SD15_MODEL_PATH}"
    target_token: str = "[V]"
    output_dir: str = "{POG_BASE_PATH}/evaluation_results"
    generated_images_dir: str = "{POG_BASE_PATH}/evaluation_generated_images"
    images_dir: str = "{POG_BASE_PATH}/images_sampled"
    captions_json: str = "{POG_BASE_PATH}/POG_captions_sampled.json"
    masked_captions_json: str = "{POG_BASE_PATH}/POG_captions_sampled_masked.json"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    image_size: int = 512
    batch_size: int = 1
    num_samples: int = None
    seed: int = 42
    use_masked_caption: bool = True


class POGEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        
        print(f"[{time.strftime('%H:%M:%S')}] Initializing POGEvaluator...")
        
        # Check CUDA availability
        print(f"[{time.strftime('%H:%M:%S')}] Checking CUDA...")
        if torch.cuda.is_available():
            print(f"[{time.strftime('%H:%M:%S')}] CUDA available: {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] CUDA not available, using CPU")
            self.device = torch.device("cpu")
        
        # Create output directories
        print(f"[{time.strftime('%H:%M:%S')}] Creating output directories...")
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.generated_images_dir, exist_ok=True)
        
        # Initialize metrics
        print(f"[{time.strftime('%H:%M:%S')}] Starting metrics initialization...")
        self._init_metrics()
        
        # Load model and embeddings
        print(f"[{time.strftime('%H:%M:%S')}] Starting model loading...")
        self._load_model()
        
        # Load test data
        print(f"[{time.strftime('%H:%M:%S')}] Loading test data...")
        self._load_test_data()
        
        # Load captions
        print(f"[{time.strftime('%H:%M:%S')}] Loading captions...")
        self._load_captions()
        
        print(f"[{time.strftime('%H:%M:%S')}] POGEvaluator initialized successfully")
    
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
        # Matching PMG_Repro exactly: data_range=2.0 (as in PMG_Repro/trainer.py line 246)
        # Note: PMG uses data_range=2.0 even though data is [0,1] from ToTensor
        self.ssim_metric = SSIM(data_range=2.0).to(self.device)  # data_range=2.0 matching PMG_Repro exactly
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
        
        # CLIP preprocessing for images in [0, 1] range (matching PMG_Repro)
        # Images are already in [0, 1] range, so no need to un-normalize
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
                    try:
                        if hasattr(hpsv2_module, '__file__'):
                            hpsv2_dir = os.path.dirname(hpsv2_module.__file__)
                        else:
                            import hpsv2
                            hpsv2_dir = os.path.dirname(hpsv2.__file__)
                        bpe_path = os.path.join(hpsv2_dir, 'src', 'open_clip', 'bpe_simple_vocab_16e6.txt.gz')
                        if not os.path.exists(bpe_path):
                            error_msg = f"HPSv2 missing required file: {bpe_path}. Please reinstall hpsv2 or download the missing file."
                            print(f"[{time.strftime('%H:%M:%S')}] Warning: {error_msg}")
                            self.hpsv2_model = None
                            self.hpsv2_failed = True
                            self.hpsv2_error_reason = error_msg
                    except Exception as test_e:
                        print(f"[{time.strftime('%H:%M:%S')}] Warning: HPSv2 file check failed: {test_e}, will try at runtime")
                        pass
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
                            print(f"[{time.strftime('%H:%M:%S')}] Warning: HPSv2 initialization failed: {e2}")
                            self.hpsv2_model = None
                            self.hpsv2_failed = True
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Warning: Failed to initialize HPSv2: {e}")
                self.hpsv2_model = None
                self.hpsv2_failed = True
        
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
        print(f"Loading embeddings from {self.config.embedding_path}...")
        embeddings = torch.load(self.config.embedding_path, map_location=self.device)
        
        # Get the embedding tensor (should be shape [8, 768])
        if self.config.target_token in embeddings:
            embedding_tensor = embeddings[self.config.target_token]
        else:
            # If key doesn't match, take the first value
            embedding_tensor = list(embeddings.values())[0]
        
        print(f"Embedding shape: {embedding_tensor.shape}")
        
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
        
        print("Model and embeddings loaded")
    
    def _load_test_data(self):
        """Load test dataset"""
        print(f"[{time.strftime('%H:%M:%S')}] Loading test data from {self.config.test_json}...")
        with open(self.config.test_json, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        if self.config.num_samples is not None:
            self.test_data = self.test_data[:self.config.num_samples]
        
        print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(self.test_data)} test samples")
    
    def _load_captions(self):
        """Load captions from JSON files"""
        print(f"[{time.strftime('%H:%M:%S')}] Loading captions...")
        
        # Load regular captions
        try:
            with open(self.config.captions_json, 'r', encoding='utf-8') as f:
                self.captions = json.load(f)
            print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(self.captions)} regular captions")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Warning: Failed to load regular captions: {e}")
            self.captions = {}
        
        # Load masked captions
        try:
            with open(self.config.masked_captions_json, 'r', encoding='utf-8') as f:
                self.masked_captions = json.load(f)
            print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(self.masked_captions)} masked captions")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Warning: Failed to load masked captions: {e}")
            self.masked_captions = {}
    
    def _get_caption(self, item_id: str) -> Optional[str]:
        """Get caption for an item_id - always use regular (complete) caption"""
        # Always use regular caption (complete description) instead of masked caption
        caption = self.captions.get(item_id)
        if caption:
            return caption
        # Fallback to masked caption if regular caption not available
        return self.masked_captions.get(item_id)
    
    def _get_user_preference_text(self, user_style: str) -> str:
        """Get user preference text from user_style string"""
        if not user_style:
            return None
        
        # user_style is a comma-separated string of keywords
        # Clean and format it
        keywords = [kw.strip() for kw in user_style.split(',') if kw.strip()]
        if not keywords:
            return None
        
        # Join keywords with commas
        preference_text = ", ".join(keywords)
        return preference_text
    
    def _image_to_tensor(self, image_path: str) -> torch.Tensor:
        """Load image and convert to tensor in [0, 1] range (matching PMG_Repro)"""
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.config.image_size, self.config.image_size), Image.BICUBIC)
        
        # Convert to tensor in [0, 1] range (matching PMG_Repro's transform)
        # PMG uses: Resize + ToTensor -> [0, 1] range
        transform = transforms.Compose([
            transforms.ToTensor()  # [0, 1] - matching PMG_Repro
        ])
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def _calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate LPIPS between two images"""
        # LPIPS expects images in [-1, 1] range, so normalize from [0,1] to [-1,1]
        # PMG uses [0,1] data but LPIPS needs [-1,1], so we normalize here
        img1_norm = (img1 * 2.0) - 1.0  # [0,1] -> [-1,1]
        img2_norm = (img2 * 2.0) - 1.0  # [0,1] -> [-1,1]
        with torch.no_grad():
            score = self.lpips_metric(img1_norm, img2_norm)
        return score.item()
    
    def _calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate SSIM between two images"""
        # Matching PMG_Repro exactly: data in [0,1] range but data_range=2.0
        # PMG uses this setting even though it seems inconsistent
        # Our images are in [0, 1] range from _image_to_tensor
        with torch.no_grad():
            score = self.ssim_metric(img1, img2)
        score_value = score.item()
        
        # Debug: Check for negative SSIM values
        if score_value < 0:
            print(f"[WARNING] SSIM calculated negative value: {score_value:.6f}")
            print(f"  Image1 range: [{img1.min().item():.4f}, {img1.max().item():.4f}]")
            print(f"  Image2 range: [{img2.min().item():.4f}, {img2.max().item():.4f}]")
            print(f"  SSIM data_range setting: 2.0")
        
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
    
    def _calculate_cps(self, image: torch.Tensor, user_style: str) -> float:
        """Calculate CPS: CLIP similarity between image and user style text"""
        preference_text = self._get_user_preference_text(user_style)
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
                if not hasattr(self, '_hpsv2_error_logged'):
                    print(f"[{time.strftime('%H:%M:%S')}] HPSv2 calculation skipped: {self.hpsv2_error_reason}")
                    self._hpsv2_error_logged = True
            return None
        
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
            if not self.hpsv2_failed:
                error_msg = f"HPSv2 calculation failed due to missing file: {e}. Please reinstall hpsv2 or download the missing file."
                print(f"[{time.strftime('%H:%M:%S')}] Error: {error_msg}")
                print(f"[{time.strftime('%H:%M:%S')}] HPSv2 will be skipped for remaining samples")
                self.hpsv2_failed = True
                self.hpsv2_error_reason = error_msg
            return None
        except Exception as e:
            error_str = str(e)
            if "Hub" in error_str or "local cache" in error_str or "Internet connection" in error_str:
                if not self.hpsv2_failed:
                    error_msg = f"HPSv2 calculation failed: Model files not found in cache. HPSv2 requires model files to be pre-downloaded. Error: {type(e).__name__}: {error_str[:200]}"
                    print(f"[{time.strftime('%H:%M:%S')}] Error: {error_msg}")
                    print(f"[{time.strftime('%H:%M:%S')}] HPSv2 will be skipped for remaining samples")
                    self.hpsv2_failed = True
                    self.hpsv2_error_reason = "Model files not found in cache (offline mode)"
                return None
            if not self.hpsv2_failed:
                error_msg = f"HPSv2 calculation failed: {type(e).__name__}: {str(e)[:200]}"
                print(f"[{time.strftime('%H:%M:%S')}] Error: {error_msg}")
                print(f"[{time.strftime('%H:%M:%S')}] HPSv2 will be skipped for remaining samples")
                import traceback
                traceback.print_exc()
                self.hpsv2_failed = True
                self.hpsv2_error_reason = error_msg
            return None
    
    def _calculate_laion_aesthetic(self, image: Image.Image) -> float:
        """
        Calculate LAION aesthetic score using CLIP + Linear Regression Head
        """
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
                import traceback
                traceback.print_exc()
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
                import traceback
                traceback.print_exc()
                self._laion_calc_error_logged = True
                if not hasattr(self, 'laion_error_reason'):
                    self.laion_error_reason = error_msg
            return None
    
    def _create_generation_prompt(self, prompt: str) -> str:
        """Create generation prompt: target description first, then token set"""
        # Format: [complete description] <v_0> ... <v_7>
        placeholder_tokens_str = " ".join([f"<v_{i}>" for i in range(8)])
        generation_prompt = f"{prompt} {placeholder_tokens_str}"
        return generation_prompt
    
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
            'laion_aesthetic': []
        }
        
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        
        for idx, sample in enumerate(tqdm(self.test_data, desc="Evaluating")):
            try:
                # Get target information
                target_item_id = sample.get('target_item_id')
                if not target_item_id:
                    print(f"Warning: No target_item_id for sample {idx}")
                    continue
                
                # Build target image path
                target_image_path = os.path.join(self.config.images_dir, f"{target_item_id}.jpg")
                if not os.path.exists(target_image_path):
                    print(f"Warning: Target image not found: {target_image_path}")
                    continue
                
                # Get caption (always use complete/regular caption, not masked)
                target_caption = self._get_caption(target_item_id)
                if not target_caption:
                    print(f"Warning: No caption available for target_item_id {target_item_id}")
                    continue
                
                # Create generation prompt: add placeholder tokens before and after the complete description
                generation_prompt = self._create_generation_prompt(target_caption)
                
                # Log prompt information (especially for first few samples)
                if idx < 3:
                    print(f"[{time.strftime('%H:%M:%S')}] Sample {idx}: Original caption: {target_caption[:100]}...")
                    print(f"[{time.strftime('%H:%M:%S')}] Sample {idx}: Generation prompt: {generation_prompt[:100]}...")
                
                # Generate image
                if idx == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Generating first image (this may take 30-60 seconds)...")
                
                # Prepare save path before generation to ensure we can save even if later steps fail
                user_id = sample.get('user_id', 'unknown')
                save_path = os.path.join(
                    self.config.generated_images_dir,
                    f"{user_id}_{target_item_id}.png"
                )
                
                # Generate and save image (ensure this happens even if later calculations fail)
                try:
                    with torch.no_grad():
                        generated_image = self.pipe(
                            generation_prompt,
                            num_inference_steps=self.config.num_inference_steps,
                            guidance_scale=self.config.guidance_scale,
                            generator=generator
                        ).images[0]
                    
                    # Immediately save the generated image
                    generated_image.save(save_path)
                    if idx == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] First image generated and saved to: {save_path}")
                    elif (idx + 1) % 50 == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] Generated and saved {idx + 1} images so far...")
                        
                except Exception as gen_e:
                    print(f"[{time.strftime('%H:%M:%S')}] ERROR: Failed to generate image for sample {idx}: {gen_e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Load target image as tensor
                target_tensor = self._image_to_tensor(target_image_path)
                generated_tensor = self._image_to_tensor(save_path)
                
                # Calculate metrics with target image
                lpips_target = self._calculate_lpips(generated_tensor, target_tensor)
                ssim_target = self._calculate_ssim(generated_tensor, target_tensor)
                
                user_style = sample.get('user_style', '')
                cps_score = self._calculate_cps(generated_tensor, user_style)
                
                # Calculate metrics with history images (up to 10)
                history_items = sample.get('history_items_info', [])[:10]
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
                    'laion_aesthetic': laion_score
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
                # Even if evaluation failed, if image was generated and saved, record it
                # (This handles cases where image generation succeeded but metric calculation failed)
                if 'save_path' in locals() and os.path.exists(save_path):
                    print(f"[{time.strftime('%H:%M:%S')}] Note: Generated image was saved at {save_path} despite evaluation error")
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
            ('laion_aesthetic', 'LAION Aesthetic')
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
                    else:
                        error_info = "N/A (model not loaded)"
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
        
        # Print summary of generated images
        print(f"\n[{time.strftime('%H:%M:%S')}] Generated images summary:")
        print(f"  Total samples processed: {len(results)}")
        print(f"  Generated images directory: {self.config.generated_images_dir}")
        
        # Count actual saved images
        if os.path.exists(self.config.generated_images_dir):
            saved_images = [f for f in os.listdir(self.config.generated_images_dir) if f.endswith('.png')]
            print(f"  Total images saved: {len(saved_images)}")
        
        # Print prompt information summary
        print(f"\n[{time.strftime('%H:%M:%S')}] Prompt information:")
        print(f"  Using complete (regular) captions, not masked captions")
        print(f"  Placeholder tokens: <v_0> <v_1> <v_2> <v_3> <v_4> <v_5> <v_6> <v_7>")
        print(f"  Prompt format: [complete description] <v_0> ... <v_7>")
        if results:
            print(f"  Example original caption: {results[0].get('prompt', 'N/A')[:100]}...")
            print(f"  Example generation prompt: {results[0].get('generation_prompt', 'N/A')[:150]}...")
        
        return avg_metrics, results


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Starting main()...", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] Python version: {sys.version}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] PyTorch version: {torch.__version__}", flush=True)
    if torch.cuda.is_available():
        print(f"[{time.strftime('%H:%M:%S')}] CUDA available: {torch.cuda.get_device_name(0)}", flush=True)
    
    parser = argparse.ArgumentParser(description="Evaluate Textual Inversion on POG dataset")
    parser.add_argument("--test_json", type=str, 
                       default="{POG_BASE_PATH}/processed_dataset/test.json",
                       help="Path to test JSON file")
    parser.add_argument("--embedding_path", type=str,
                       default="{POG_BASE_PATH}/textual_inversion_sd15/learned_embeds.bin",
                       help="Path to learned embeddings")
    parser.add_argument("--sd15_path", type=str,
                       default="{SD15_MODEL_PATH}",
                       help="Path to Stable Diffusion 1.5 model")
    parser.add_argument("--output_dir", type=str,
                       default="{POG_BASE_PATH}/evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--generated_images_dir", type=str,
                       default="{POG_BASE_PATH}/evaluation_generated_images",
                       help="Directory to save generated images")
    parser.add_argument("--images_dir", type=str,
                       default="{POG_BASE_PATH}/images_sampled",
                       help="Directory containing images")
    parser.add_argument("--captions_json", type=str,
                       default="{POG_BASE_PATH}/POG_captions_sampled.json",
                       help="Path to captions JSON file")
    parser.add_argument("--masked_captions_json", type=str,
                       default="{POG_BASE_PATH}/POG_captions_sampled_masked.json",
                       help="Path to masked captions JSON file")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (None for all)")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--use_masked_caption", action='store_true',
                       help="[DEPRECATED] Always uses regular (complete) captions now. This option is ignored.")
    parser.add_argument("--use_regular_caption", action='store_true',
                       help="[DEPRECATED] Always uses regular (complete) captions now. This option is ignored.")
    
    args = parser.parse_args()
    
    # Always use regular (complete) captions, not masked captions
    # The prompt format is: <v_0> ... <v_7> [complete description] <v_0> ... <v_7>
    
    config = EvalConfig(
        test_json=args.test_json,
        embedding_path=args.embedding_path,
        sd15_path=args.sd15_path,
        output_dir=args.output_dir,
        generated_images_dir=args.generated_images_dir,
        images_dir=args.images_dir,
        captions_json=args.captions_json,
        masked_captions_json=args.masked_captions_json,
        num_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        use_masked_caption=False  # Always False - always use complete captions
    )
    
    try:
        print(f"[{time.strftime('%H:%M:%S')}] Creating POGEvaluator...", flush=True)
        evaluator = POGEvaluator(config)
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

