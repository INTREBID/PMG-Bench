# Personalized_Generation/calculate_metrics.py
import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
import argparse
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 指标导入
import lpips
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

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


class MetricsEvaluator:
    """评估指标计算器（使用本地CLIP模型）"""
    
    def __init__(self, device: str = "cuda", image_size: int = 512, user_preferences_path: str = None):
        self.device = torch.device(device)
        self.image_size = image_size
        
        # 本地CLIP模型路径
        self.local_clip_path = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/clip-vit-base-patch32"
        
        # 初始化指标
        print("Initializing metrics...")
        self._init_metrics()
        
        # 加载用户偏好（用于CPS计算）
        self._load_user_preferences(user_preferences_path)
    
    def _init_metrics(self):
        """初始化所有评估指标"""
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
        
        # CLIP (从本地路径加载)
        try:
            if not os.path.exists(self.local_clip_path):
                print(f"Error: Local CLIP model not found at {self.local_clip_path}")
                self.clip_model = None
                self.clip_processor = None
            else:
                print(f"Loading CLIP model from local path: {self.local_clip_path}")
                self.clip_processor = CLIPProcessor.from_pretrained(
                    self.local_clip_path,
                    local_files_only=True
                )
                self.clip_model = CLIPModel.from_pretrained(
                    self.local_clip_path,
                    local_files_only=True
                ).to(self.device)
                self.clip_model.eval()
                
                # CLIP transform for tensor images in [0, 1] range
                self.clip_transform = transforms.Compose([
                    transforms.Resize(224, antialias=True),
                    transforms.CenterCrop(224),
                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                       std=(0.26862954, 0.26130258, 0.27577711))
                ])
                print("CLIP initialized from local path")
        except Exception as e:
            print(f"Error initializing CLIP: {e}")
            self.clip_model = None
            self.clip_processor = None
        
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
        
        # LAION Aesthetic (使用同一个本地CLIP模型)
        self.laion_model = self.clip_model  # 复用同一个CLIP模型
        self.laion_processor = self.clip_processor
        self._laion_loaded = (self.clip_model is not None)
        
        print("Metrics initialization complete")
    
    def _load_user_preferences(self, preferences_path: str = None):
        """加载用户偏好（用于CPS计算）"""
        if preferences_path is None:
            self.user_preferences = {}
            return
        
        try:
            if os.path.exists(preferences_path):
                with open(preferences_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 根据不同的数据格式处理
                if isinstance(data, list):
                    # FLICKR和POG格式：数组，每个元素有worker/user和style字段
                    self.user_preferences = {}
                    for item in data:
                        # FLICKR使用'worker'，POG使用'user'
                        user_id = item.get('worker') or item.get('user')
                        style = item.get('style', '')
                        if user_id:
                            self.user_preferences[user_id] = style
                    print(f"Loaded preferences for {len(self.user_preferences)} users (list format)")
                elif isinstance(data, dict):
                    # SER格式：对象，键是topic，值包含keywords
                    self.user_preferences = data
                    print(f"Loaded preferences for {len(self.user_preferences)} users (dict format)")
            else:
                print(f"Warning: Preferences file not found: {preferences_path}")
                self.user_preferences = {}
        except Exception as e:
            print(f"Warning: Failed to load user preferences: {e}")
            self.user_preferences = {}
    
    def _get_user_preference_text(self, history_topic: str) -> Optional[str]:
        """获取用户偏好文本"""
        if history_topic not in self.user_preferences:
            return None
        
        user_pref = self.user_preferences[history_topic]
        
        # 处理不同的格式
        if isinstance(user_pref, str):
            # FLICKR/POG格式：直接是style字符串
            return user_pref if user_pref else None
        elif isinstance(user_pref, dict):
            # SER格式：从keywords提取
            keywords = user_pref.get('keywords', [])
            if not keywords:
                return None
            return ", ".join(keywords)
        
        return None
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """将PIL图像转换为tensor [0, 1]"""
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        transform = transforms.Compose([
            transforms.ToTensor()  # [0, 1]
        ])
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """计算LPIPS"""
        if self.lpips_metric is None:
            return None
        # LPIPS expects images in [-1, 1] range
        img1_norm = (img1 * 2.0) - 1.0  # [0,1] -> [-1,1]
        img2_norm = (img2 * 2.0) - 1.0  # [0,1] -> [-1,1]
        with torch.no_grad():
            score = self.lpips_metric(img1_norm, img2_norm)
        return score.item()
    
    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """计算SSIM"""
        if self.ssim_metric is None:
            return None
        with torch.no_grad():
            score = self.ssim_metric(img1, img2)
        return score.item()
    
    def calculate_cps(self, image_tensor: torch.Tensor, history_topic: str) -> Optional[float]:
        """计算CPS: CLIP图像与用户偏好文本的相似度"""
        if self.clip_model is None or self.clip_processor is None:
            return None
        preference_text = self._get_user_preference_text(history_topic)
        if preference_text is None:
            return None
        
        with torch.no_grad():
            # 将tensor转换为PIL Image
            image_pil = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())
            
            # 使用processor处理图像和文本
            inputs = self.clip_processor(
                text=[preference_text],
                images=image_pil,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # 获取特征
            outputs = self.clip_model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = (image_features @ text_features.T).item()
        return similarity
    
    def calculate_cpis(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """计算CPIS: CLIP图像-图像相似度"""
        if self.clip_model is None or self.clip_processor is None:
            return None
        
        with torch.no_grad():
            # 将tensor转换为PIL Image
            img1_pil = transforms.ToPILImage()(img1.squeeze(0).cpu())
            img2_pil = transforms.ToPILImage()(img2.squeeze(0).cpu())
            
            # 分别处理两张图像
            inputs1 = self.clip_processor(images=img1_pil, return_tensors="pt").to(self.device)
            inputs2 = self.clip_processor(images=img2_pil, return_tensors="pt").to(self.device)
            
            # 获取图像特征
            features1 = self.clip_model.get_image_features(**inputs1)
            features2 = self.clip_model.get_image_features(**inputs2)
            
            # Normalize
            features1 = features1 / features1.norm(dim=-1, keepdim=True)
            features2 = features2 / features2.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = (features1 @ features2.T).item()
        return similarity
    
    def calculate_hpsv2(self, image_path: str, prompt: str) -> Optional[float]:
        """计算HPSv2"""
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
        """计算LAION美学评分（使用本地CLIP模型）"""
        if self.laion_model is None or self.laion_processor is None:
            return None
        
        try:
            with torch.no_grad():
                # 使用CLIP processor处理图像
                inputs = self.laion_processor(images=image, return_tensors="pt").to(self.device)
                
                # 获取图像特征
                image_features = self.laion_model.get_image_features(**inputs)
                
                # 简单的美学评分：使用特征的L2范数作为代理指标
                # 注意：这不是真正的LAION美学评分，需要训练好的线性层
                # 如果有预训练的美学预测头，应该加载并使用它
                aesthetic_score = image_features.norm(dim=-1).mean().item()
                
            return aesthetic_score
        except Exception as e:
            print(f"Warning: LAION aesthetic calculation failed: {e}")
            return None
    
    def evaluate_generated_image(
        self,
        generated_image: Image.Image,
        target_image_path: str,
        history_images: List[Image.Image],
        history_topic: str,
        target_caption: str,
        generated_image_path: str
    ) -> Dict[str, Optional[float]]:
        """评估生成的图像，返回所有指标"""
        metrics = {}
        
        # 转换为tensor
        generated_tensor = self._image_to_tensor(generated_image)
        
        # 加载目标图像
        try:
            target_image = Image.open(target_image_path).convert("RGB")
            target_tensor = self._image_to_tensor(target_image)
        except Exception as e:
            print(f"Warning: Failed to load target image: {e}")
            target_tensor = None
        
        # LPIPS vs Target & SSIM vs Target
        if target_tensor is not None:
            metrics['lpips_target'] = self.calculate_lpips(generated_tensor, target_tensor)
            metrics['ssim_target'] = self.calculate_ssim(generated_tensor, target_tensor)
        else:
            metrics['lpips_target'] = None
            metrics['ssim_target'] = None
        
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
        
        metrics['lpips_history_avg'] = float(np.mean(lpips_history_scores)) if lpips_history_scores else None
        metrics['ssim_history_avg'] = float(np.mean(ssim_history_scores)) if ssim_history_scores else None
        metrics['cpis_history_avg'] = float(np.mean(cpis_history_scores)) if cpis_history_scores else None
        
        # CPS
        metrics['cps'] = self.calculate_cps(generated_tensor, history_topic)
        
        # HPSv2
        metrics['hpsv2'] = self.calculate_hpsv2(generated_image_path, target_caption)
        
        # LAION Aesthetic
        metrics['laion_aesthetic'] = self.calculate_laion_aesthetic(generated_image)
        
        return metrics


def evaluate_dataset(
    dataset_name: str,
    test_json_path: str,
    eval_output_dir: str,
    output_json_path: str,
    user_preferences_path: str = None,
    device='cuda'
):
    """评估整个数据集"""
    print(f"\n{'='*60}")
    print(f"Evaluating {dataset_name} Dataset")
    print(f"{'='*60}\n")
    
    # 加载测试数据
    print(f"Loading test data from {test_json_path}...")
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples\n")
    
    # 初始化指标计算器
    evaluator = MetricsEvaluator(
        device=device,
        image_size=512,
        user_preferences_path=user_preferences_path
    )
    
    # 评估每个样本
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
    
    for idx, sample in enumerate(tqdm(test_data, desc="Calculating metrics")):
        sample_dir = os.path.join(eval_output_dir, f"sample_{idx:04d}")
        generated_image_path = os.path.join(sample_dir, "gen_0.jpg")
        
        # 检查生成图像是否存在
        if not os.path.exists(generated_image_path):
            print(f"Warning: Generated image not found for sample {idx}: {generated_image_path}")
            continue
        
        # 加载生成图像
        try:
            generated_image = Image.open(generated_image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to load generated image for sample {idx}: {e}")
            continue
        
        # 获取目标和历史图像路径（修正：从 target_item_info 获取）
        target_item_info = sample.get('target_item_info', {})
        target_image_path = target_item_info.get('image_path')
        target_caption = target_item_info.get('caption', '')
        
        # 检查 target_image_path 是否存在
        if not target_image_path:
            print(f"Warning: No target image path for sample {idx}")
            continue
        
        # 获取历史图像路径
        history_image_paths = [item['image_path'] for item in sample['history_items_info']]
        
        # 加载历史图像
        history_images = []
        for hist_path in history_image_paths:
            try:
                if os.path.exists(hist_path):
                    hist_img = Image.open(hist_path).convert("RGB")
                    history_images.append(hist_img)
            except Exception as e:
                print(f"Warning: Failed to load history image {hist_path}: {e}")
        
        if len(history_images) == 0:
            print(f"Warning: No valid history images for sample {idx}")
            continue
        
        # 获取history_topic（用于CPS）
        history_topic = sample.get('user_id') or sample.get('worker_id', '')
        
        # 计算指标
        metrics = evaluator.evaluate_generated_image(
            generated_image=generated_image,
            target_image_path=target_image_path,
            history_images=history_images,
            history_topic=history_topic,
            target_caption=target_caption,
            generated_image_path=generated_image_path
        )
        
        # 保存结果
        result = {
            'sample_idx': idx,
            'user_id': sample.get('user_id') or sample.get('worker_id'),
            'target_item_id': sample['target_item_id'],
            'metrics': metrics
        }
        results.append(result)
        
        # 累积指标
        for key in all_metrics.keys():
            if metrics.get(key) is not None:
                all_metrics[key].append(metrics[key])
    
    # 计算平均指标和标准差
    avg_metrics = {}
    std_metrics = {}
    for key, values in all_metrics.items():
        if values:
            avg_metrics[key] = float(np.mean(values))
            std_metrics[key] = float(np.std(values))
        else:
            avg_metrics[key] = None
            std_metrics[key] = None
    
    # 保存结果
    output_data = {
        'dataset': dataset_name,
        'total_samples': len(test_data),
        'evaluated_samples': len(results),
        'average_metrics': avg_metrics,
        'std_metrics': std_metrics,
        'per_sample_results': results
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"Evaluation Results for {dataset_name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(test_data)}")
    print(f"Evaluated samples: {len(results)}")
    print(f"\nAverage Metrics:")
    print(f"  {'Metric':<25s} {'Mean':<12s} {'Std':<12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    for key in ['lpips_target', 'lpips_history_avg', 'ssim_target', 'ssim_history_avg', 
                'cps', 'cpis_history_avg', 'hpsv2', 'laion_aesthetic']:
        mean_val = avg_metrics.get(key)
        std_val = std_metrics.get(key)
        if mean_val is not None:
            print(f"  {key:<25s} {mean_val:<12.6f} {std_val:<12.6f}")
        else:
            print(f"  {key:<25s} {'N/A':<12s} {'N/A':<12s}")
    print(f"\nResults saved to {output_json_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="计算生成图像的评估指标")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['FLICKR', 'POG', 'SER'],
        help="数据集名称"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="计算设备"
    )
    
    args = parser.parse_args()
    
    # 根据数据集设置路径
    if args.dataset == 'FLICKR':
        test_json_path = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/FLICKR/processed_dataset/test.json"
        eval_output_dir = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/FLICKR/eval_outputs"
        output_json_path = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/FLICKR/eval_outputs/metrics_results.json"
        user_preferences_path = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/FLICKR/FLICKR_styles.json"
    elif args.dataset == 'POG':
        test_json_path = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/POG/processed_dataset/test.json"
        eval_output_dir = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/POG/eval_outputs"
        output_json_path = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/POG/eval_outputs/metrics_results.json"
        user_preferences_path = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/POG/user_styles.json"
    elif args.dataset == 'SER':
        test_json_path = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/SER_Dataset/processed_dataset/test.json"
        eval_output_dir = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/SER_Dataset/eval_outputs"
        output_json_path = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/SER_Dataset/eval_outputs/metrics_results.json"
        user_preferences_path = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/SER_Dataset/user_preferences.json"
    
    # 评估数据集
    evaluate_dataset(
        dataset_name=args.dataset,
        test_json_path=test_json_path,
        eval_output_dir=eval_output_dir,
        output_json_path=output_json_path,
        user_preferences_path=user_preferences_path,
        device=args.device
    )


if __name__ == "__main__":
    main()