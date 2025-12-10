
import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
import argparse
from datetime import datetime

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
    
    def __init__(self, device: str = "cuda", image_size: int = 512):
        self.device = torch.device(device)
        self.image_size = image_size
        
        print("Initializing metrics...")
        self._init_metrics()
    
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
            # Matching PMG_Repro exactly: data_range=2.0 (as in PMG_Repro/trainer.py line 246)
            # Note: PMG uses data_range=2.0 even though data is [0,1] from ToTensor
            self.ssim_metric = SSIM(data_range=2.0).to(self.device)  # data_range=2.0 matching PMG_Repro exactly
            print("SSIM initialized")
        except Exception as e:
            print(f"Error initializing SSIM: {e}")
            self.ssim_metric = None
        
        # CLIP
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            # CLIP transform for tensor images in [0, 1] range (matching PMG_Repro)
            # Images are already in [0, 1] range, so no need to un-normalize
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
        # Convert to tensor in [0, 1] range (matching PMG_Repro's transform)
        # PMG uses: Resize + ToTensor -> [0, 1] range
        transform = transforms.Compose([
            transforms.ToTensor()  # [0, 1] - matching PMG_Repro
        ])
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """Calculate LPIPS"""
        if self.lpips_metric is None:
            return None
        # LPIPS expects images in [-1, 1] range, so normalize from [0,1] to [-1,1]
        # PMG uses [0,1] data but LPIPS needs [-1,1], so we normalize here
        img1_norm = (img1 * 2.0) - 1.0  # [0,1] -> [-1,1]
        img2_norm = (img2 * 2.0) - 1.0  # [0,1] -> [-1,1]
        with torch.no_grad():
            score = self.lpips_metric(img1_norm, img2_norm)
        return score.item()
    
    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """Calculate SSIM"""
        if self.ssim_metric is None:
            return None
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
    
    def _get_user_preference_text(self, user_style: str) -> Optional[str]:
        """Get user preference text from user_style string"""
        if not user_style:
            return None
        
        lines = user_style.split('\n')
        keywords = []
        for line in lines:
            line_keywords = [kw.strip() for kw in line.split(',') if kw.strip()]
            keywords.extend(line_keywords)
        
        if not keywords:
            return None
        
        unique_keywords = list(dict.fromkeys(keywords))
        preference_text = ", ".join(unique_keywords)
        return preference_text
    
    def calculate_cps(self, image_tensor: torch.Tensor, user_style: str) -> Optional[float]:
        """Calculate CPS: CLIP image-text similarity"""
        if self.clip_model is None:
            return None
        preference_text = self._get_user_preference_text(user_style)
        if preference_text is None:
            return None
        
        with torch.no_grad():
            # Preprocess image for CLIP (image_tensor is in [0, 1] range)
            # CLIP transform expects [0, 1] input, which matches our data range
            image_clip = self.clip_transform(image_tensor)
            # Encode image and text
            image_features = self.clip_model.encode_image(image_clip)
            text_tokens = clip.tokenize([preference_text]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Cosine similarity
            similarity = (image_features @ text_features.T).item()
        return similarity
    
    def calculate_cpis(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """Calculate CPIS: CLIP image-image similarity"""
        if self.clip_model is None:
            return None
        with torch.no_grad():
            # Both images are in [0, 1] range, CLIP transform expects [0, 1] input
            img1_clip = self.clip_transform(img1)
            img2_clip = self.clip_transform(img2)
            features1 = self.clip_model.encode_image(img1_clip)
            features2 = self.clip_model.encode_image(img2_clip)
            # Normalize
            features1 = features1 / features1.norm(dim=-1, keepdim=True)
            features2 = features2 / features2.norm(dim=-1, keepdim=True)
            # Cosine similarity
            similarity = (features1 @ features2.T).item()
        return similarity
    
    def calculate_hpsv2(self, image_path: str, prompt: str) -> Optional[float]:
        """Calculate HPSv2"""
        if self.hpsv2_model is None or self.hpsv2_failed:
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
            
            # Handle different return types
            if isinstance(result, (list, tuple)):
                return float(result[0])
            elif isinstance(result, torch.Tensor):
                return float(result.item())
            elif isinstance(result, np.ndarray):
                return float(result.item() if result.size == 1 else result[0])
            else:
                return float(result)
        except Exception as e:
            if not self.hpsv2_failed:
                print(f"Warning: HPSv2 calculation failed: {e}")
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

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                norm_value = image_features.norm(dim=-1).item()
                if abs(norm_value - 1.0) > 0.01:
                    print(f"[ERROR] LAION normalization failed! Norm value: {norm_value:.6f} (expected ~1.0)")
                    print(f"[ERROR] This will cause score explosion. Stopping immediately.")
                    raise ValueError(f"Normalization failed: norm={norm_value:.6f}")

                image_features = image_features.float()

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
        user_style: str,
        target_caption: str,
        generated_image_path: str
    ) -> Dict[str, Optional[float]]:
        """Evaluate generated image and return all metrics"""
        metrics = {}
        
        generated_tensor = self._image_to_tensor(generated_image)
        try:
            target_image = Image.open(target_image_path).convert("RGB")
            target_tensor = self._image_to_tensor(target_image)
        except Exception as e:
            print(f"Warning: Failed to load target image: {e}")
            target_tensor = None
        
        # LPIPS vs Target
        if target_tensor is not None:
            metrics['lpips_target'] = self.calculate_lpips(generated_tensor, target_tensor)
            metrics['ssim_target'] = self.calculate_ssim(generated_tensor, target_tensor)
        
        # LPIPS vs History Avg, SSIM vs History Avg, CPIS vs History Avg
        lpips_history_scores = []
        ssim_history_scores = []
        cpis_history_scores = []
        
        for hist_img in history_images:
            try:
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
                print(f"Warning: Failed to process history image: {e}")
                continue
        
        metrics['lpips_history_avg'] = np.mean(lpips_history_scores) if lpips_history_scores else None
        metrics['ssim_history_avg'] = np.mean(ssim_history_scores) if ssim_history_scores else None
        metrics['cpis_history_avg'] = np.mean(cpis_history_scores) if cpis_history_scores else None
        
        metrics['cps'] = self.calculate_cps(generated_tensor, user_style)
        
        # HPSv2
        metrics['hpsv2'] = self.calculate_hpsv2(generated_image_path, target_caption)
        
        # LAION Aesthetic
        metrics['laion_aesthetic'] = self.calculate_laion_aesthetic(generated_image)
        
        return metrics


def load_images(image_paths: List[str]) -> List[Image.Image]:
    """Load list of images"""
    images = []
    for img_path in image_paths:
        try:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            else:
                print(f"Warning: Image not found: {img_path}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
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
    """Load multiple IP-Adapter instances for multi-image input"""
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
    guidance_scale: float = 10.0,
    ip_adapter_scale: float = 0.3,
    seed: Optional[int] = None
) -> Image.Image:
    """Generate personalized image"""
    if prompt is None:
        prompt = target_caption
    
    ip_adapter_images = history_images
    
    num_loaded = 1
    if hasattr(pipe.unet, 'encoder_hid_proj') and hasattr(pipe.unet.encoder_hid_proj, 'image_projection_layers'):
        num_loaded = len(pipe.unet.encoder_hid_proj.image_projection_layers)
        if num_loaded < len(history_images):
            print(f"Warning: Need {len(history_images)} IP-Adapters but only {num_loaded} loaded. Using first {num_loaded} images.")
            ip_adapter_images = history_images[:num_loaded]
        elif num_loaded > len(history_images):
            while len(ip_adapter_images) < num_loaded:
                ip_adapter_images.append(history_images[-1])
    
    if num_loaded > 1:
        pipe.set_ip_adapter_scale([ip_adapter_scale] * num_loaded)
    else:
        pipe.set_ip_adapter_scale(ip_adapter_scale)
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    if not prompt or len(prompt.strip()) == 0:
        print(f"[WARNING] Empty prompt detected! Using default prompt.")
        prompt = "a photo"
    
    if hasattr(generate_personalized_image, '_debug_count'):
        generate_personalized_image._debug_count += 1
    else:
        generate_personalized_image._debug_count = 1
    
    if generate_personalized_image._debug_count <= 3:
        print(f"[DEBUG] generate_personalized_image call #{generate_personalized_image._debug_count}")
        print(f"[DEBUG]   prompt: '{prompt[:100]}...' (length: {len(prompt)})")
        print(f"[DEBUG]   num_ip_adapter_images: {len(ip_adapter_images)}")
        print(f"[DEBUG]   ip_adapter_scale: {ip_adapter_scale}")
        print(f"[DEBUG]   guidance_scale: {guidance_scale}")
    
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=ip_adapter_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
    
    return result.images[0]


def process_dataset(
    test_json_path: str,
    output_dir: str,
    sd_model_path: str,
    ip_adapter_path: str,
    ip_adapter_weight: str,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    start_idx: int = 0,
    num_inference_steps: int = 50,
    guidance_scale: float = 10.0,
    ip_adapter_scale: float = 0.3,
    save_history_images: bool = False,
    seed: int = 42
):
    """Process entire POG dataset"""
    os.makedirs(output_dir, exist_ok=True)
    generated_dir = os.path.join(output_dir, "generated_images")
    os.makedirs(generated_dir, exist_ok=True)
    
    if save_history_images:
        history_dir = os.path.join(output_dir, "history_images")
        os.makedirs(history_dir, exist_ok=True)
    
    print(f"Loading test dataset from {test_json_path}...")
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
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
    evaluator = MetricsEvaluator(device=device, image_size=512)
    
    results = []
    failed_samples = []
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
    
    for idx, sample in enumerate(tqdm(test_data, desc="Generating images")):
        try:
            history_image_paths = [item['image_path'] for item in sample.get('history_items_info', []) 
                                  if item.get('image_path')]
            
            MAX_HISTORY_IMAGES = 10
            if len(history_image_paths) > MAX_HISTORY_IMAGES:
                history_image_paths = history_image_paths[-MAX_HISTORY_IMAGES:]
                if idx < 3:
                    print(f"[DEBUG] Sample {idx}: Limited history images from {len(sample.get('history_items_info', []))} to {len(history_image_paths)} (last {MAX_HISTORY_IMAGES})")
            
            history_images = load_images(history_image_paths)
            
            if len(history_images) == 0:
                print(f"Warning: No valid history images for sample {idx}")
                failed_samples.append({
                    'index': idx,
                    'reason': 'No valid history images'
                })
                continue
            
            if len(history_images) > 1:
                num_loaded = 1
                if hasattr(pipe.unet, 'encoder_hid_proj') and hasattr(pipe.unet.encoder_hid_proj, 'image_projection_layers'):
                    num_loaded = len(pipe.unet.encoder_hid_proj.image_projection_layers)
                
                if num_loaded < len(history_images):
                    load_multiple_ip_adapters(
                        pipe,
                        ip_adapter_path,
                        ip_adapter_weight,
                        len(history_images)
                    )
            
            target_item_info = sample.get('target_item_info', {})
            target_caption = target_item_info.get('caption', '')
            target_image_path = target_item_info.get('image_path')
            target_item_id = sample.get('target_item_id', f'sample_{idx}')
            
            if not target_caption:
                print(f"Warning: No target caption for sample {idx}")
                failed_samples.append({
                    'index': idx,
                    'reason': 'No target caption'
                })
                continue
            
            if idx < 3:
                print(f"[DEBUG] Sample {idx}: target_caption = '{target_caption}'")
                print(f"[DEBUG] Sample {idx}: num_history_images = {len(history_images)}")
            
            generated_image = generate_personalized_image(
                pipe=pipe,
                history_images=history_images,
                target_caption=target_caption,
                target_image_path=target_image_path,
                prompt=target_caption,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                ip_adapter_scale=ip_adapter_scale,
                seed=seed
            )
            
            user_id = sample.get('user_id', 'unknown')
            safe_filename = f"{user_id}_{target_item_id}".replace('/', '_').replace('\\', '_')
            output_path = os.path.join(generated_dir, f"{safe_filename}_generated.png")
            generated_image.save(output_path)
            
            if save_history_images:
                for i, hist_img in enumerate(history_images):
                    hist_output_path = os.path.join(history_dir, f"{safe_filename}_history_{i}.png")
                    hist_img.save(hist_output_path)
            
            print(f"Calculating metrics for sample {idx}...")
            user_style = sample.get('user_style', '')
            metrics = evaluator.evaluate_generated_image(
                generated_image=generated_image,
                target_image_path=target_image_path,
                history_images=history_images,
                user_style=user_style,
                target_caption=target_caption,
                generated_image_path=output_path
            )
            
            result = {
                'index': idx + start_idx,
                'user_id': user_id,
                'target_item_id': target_item_id,
                'num_history_images': len(history_images),
                'target_caption': target_caption,
                'output_path': output_path,
                'target_image_path': target_image_path,
                'success': True,
                'metrics': metrics
            }
            results.append(result)
            
            for key in all_metrics.keys():
                if metrics.get(key) is not None:
                    all_metrics[key].append(metrics[key])
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            failed_samples.append({
                'index': idx,
                'reason': str(e)
            })
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
    print(f"Generation and Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {len(test_data)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed_samples)}")
    print(f"\nAverage Metrics:")
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
    parser = argparse.ArgumentParser(description="POG dataset personalized image generation script")
    parser.add_argument(
        "--test_json",
        type=str,
        default="{POG_BASE_PATH}/processed_dataset/test.json",
        help="Test JSON path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="{POG_BASE_PATH}/ip_adapter_generated",
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
        "--save_history_images",
        action="store_true",
        help="Save history images"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"
    
    process_dataset(
        test_json_path=args.test_json,
        output_dir=args.output_dir,
        sd_model_path=args.sd_model_path,
        ip_adapter_path=args.ip_adapter_path,
        ip_adapter_weight=args.ip_adapter_weight,
        device=args.device,
        max_samples=args.max_samples,
        start_idx=args.start_idx,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        ip_adapter_scale=args.ip_adapter_scale,
        save_history_images=args.save_history_images,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

