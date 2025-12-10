#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLICKR_PMG_INFER.py - FLICKR数据集评估脚本（三路条件融合，使用worker_styles）
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ====== 配置参数 ======
class Args:
    num_image_prompt = 2
    num_prefix_prompt = 2
    max_txt_len = 600
    image_size = 512
    weight_dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_path = '/data-nfs/gpu1-1/ud202581869/Personalized_Generation/FLICKR/processed_dataset'
    styles_path = '/data-nfs/gpu1-1/ud202581869/Personalized_Generation/FLICKR/FLICKR_styles.json'
    llama_path = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/Llama2_7b"
    sd_path = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/stable-diffusion-v1-5"
    checkpoint_path = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/FLICKR/train_result_1/logs/flickr_aesthetic/save-steps-4000.pth"  # 指定训练好的checkpoint路径，例如："/path/to/save-steps-3000.pth"
    output_dir = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/FLICKR/eval_outputs"
    
    # 生成参数
    num_images_per_sample = 1
    seed = 42
    neg_prompt = "lowres, text, error, cropped, worst quality, low quality"
    
    # 三路权重（可调节）
    weight_target = 1      # target caption 权重
    weight_image = 1       # LLaMA image prompt 权重
    weight_preference = 1  # user style/preference 权重

args = Args()

# ====== 加载用户风格 ======
print("=" * 80)
print("Loading worker styles...")
print("=" * 80)

with open(args.styles_path, 'r', encoding='utf-8') as f:
    worker_styles_list = json.load(f)

# 转换为字典格式 {worker_id: style_text}
worker_styles = {}
for item in worker_styles_list:
    worker_id = str(item['worker'])  # 确保是字符串
    style_text = item.get('style', '')
    worker_styles[worker_id] = style_text

print(f"Loaded styles for {len(worker_styles)} workers")

# ====== 加载测试数据 ======
print("\nLoading FLICKR test data...")
test_json_path = os.path.join(args.data_path, 'test.json')
with open(test_json_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
print(f"Loaded {len(test_data)} test samples")

# ====== 工具函数 ======
def _resize_rgb(img_path, size):
    if img_path is None or not os.path.exists(img_path):
        return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    try:
        im = Image.open(img_path).convert('RGB')
        im = im.resize((size, size), Image.BICUBIC)
        return np.ascontiguousarray(np.array(im, dtype=np.uint8))
    except:
        return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

def save_image_grid(images, save_path):
    if len(images) == 0:
        return
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    img_h, img_w = images[0].shape[:2]
    grid = np.ones((rows * img_h, cols * img_w, 3), dtype=np.uint8) * 255
    for idx, img in enumerate(images):
        row, col = idx // cols, idx % cols
        grid[row*img_h:(row+1)*img_h, col*img_w:(col+1)*img_w] = img
    Image.fromarray(grid).save(save_path)

def get_worker_style_text(worker_id):
    """从worker_styles中获取用户风格文本"""
    worker_id_str = str(worker_id)
    if worker_id_str in worker_styles:
        style_text = worker_styles[worker_id_str]
        if style_text:
            return style_text + "aesthetic visual style, high quality photography"
    return "aesthetic visual style, high quality photography"

# ====== 加载 LLaMA ======
print("\nLoading LLaMA model...")
from transformers import LlamaForCausalLM, LlamaTokenizer

llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_path)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_model = LlamaForCausalLM.from_pretrained(
    args.llama_path, torch_dtype=args.weight_dtype,
    low_cpu_mem_usage=True, device_map="auto"
)
llama_model.eval()

# ====== 加载 SD Pipeline ======
print("Loading Stable Diffusion pipeline...")
from myCustomPipeline import SDPipeline

sd_pipeline = SDPipeline(args.weight_dtype, model_id=args.sd_path)
sd_pipeline = sd_pipeline.to(args.device)

# ====== Prompt 处理 ======
def prompt_preprocess(history_captions):
    """构造 LLaMA prompt（FLICKR 风格，不包含 style）"""
    prompt = (
        "### Human: A person rated the following images highly: \"<Images/>\". Describe their visual taste. ###Assistant: "
    )
    prompt = prompt.replace('<Images/>', history_captions)
    return prompt

# ====== InferenceModel ======
class PrefixEncoder(torch.nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, pre_seq_len, prefix_projection=False, prefix_hidden_size=4096):
        super().__init__()
        self.prefix_projection = prefix_projection
        if self.prefix_projection:
            self.embedding = torch.nn.Embedding(pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(prefix_hidden_size, num_hidden_layers * 2 * hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(pre_seq_len, num_hidden_layers * 2 * hidden_size)

    def forward(self, prefix):
        if self.prefix_projection:
            past_key_values = self.trans(self.embedding(prefix))
        else:
            past_key_values = self.embedding(prefix)
        # 确保输出与模型 dtype 一致
        return past_key_values.to(torch.bfloat16)


class InferenceModel(torch.nn.Module):
    def __init__(self, layer_num, num_image_prompt, num_prefix_prompt, emb_dim, sd_hidden_state_dim):
        super().__init__()
        self.layer_num = layer_num
        self.num_image_prompt = num_image_prompt
        self.num_prefix_prompt = num_prefix_prompt
        self.emb_dim = emb_dim
        self.mapping_layer = torch.nn.Linear(emb_dim, sd_hidden_state_dim)
        self.trainable_prompt = torch.nn.Parameter(torch.randn((1, num_image_prompt, emb_dim), requires_grad=True))
        self.prefix_tokens = torch.arange(num_prefix_prompt).long()
        self.prefix_encoder = PrefixEncoder(layer_num, 4096, num_prefix_prompt)

    def forward(self, llama_model, token, token_len):
        bsz = token.shape[0]
        attention_mask = token != llama_tokenizer.pad_token_id
        emb = llama_model.model.embed_tokens(token)
        for i in range(bsz):
            l = token_len[i].item()
            emb[i, l:l+self.num_image_prompt] = self.trainable_prompt
            attention_mask[i, l:l+self.num_image_prompt] = 1
        attention_mask = torch.concat([
            torch.ones((bsz, self.num_prefix_prompt), device=attention_mask.device),
            attention_mask
        ], dim=1)

        num_head = llama_model.model.layers[0].self_attn.num_heads
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(token.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(bsz, self.num_prefix_prompt, self.layer_num, 2, num_head, -1)
        past_key_values = past_key_values.permute(2, 3, 0, 4, 1, 5)

        outputs = llama_model.model.forward(
            inputs_embeds=emb, output_hidden_states=True,
            attention_mask=attention_mask, past_key_values=past_key_values,
        )
        encoder_hidden_states = [
            outputs.last_hidden_state[i, token_len[i]:token_len[i]+self.num_image_prompt]
            for i in range(bsz)
        ]
        return self.mapping_layer(torch.stack(encoder_hidden_states))

# ====== 加载推理模型 ======
print("Loading inference model...")
infer_model = InferenceModel(
    layer_num=len(llama_model.model.layers),
    num_image_prompt=args.num_image_prompt,
    num_prefix_prompt=args.num_prefix_prompt,
    emb_dim=4096, sd_hidden_state_dim=768
).to(args.device)

if args.checkpoint_path and os.path.exists(args.checkpoint_path):
    state = torch.load(args.checkpoint_path, map_location="cuda")
    infer_model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {args.checkpoint_path}")
else:
    print("Warning: No checkpoint loaded!")

# 确保模型所有参数都是 bfloat16
infer_model = infer_model.to(args.weight_dtype)
print(f"Model converted to dtype: {args.weight_dtype}")

infer_model.eval()
os.makedirs(args.output_dir, exist_ok=True)

# ====== 评估循环 ======
print("\n" + "=" * 80)
print(f"Starting evaluation (3-way conditioning with worker styles)...")
print(f"Weights: target={args.weight_target}, image={args.weight_image}, preference={args.weight_preference}")
print("=" * 80)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

for idx, sample in enumerate(tqdm(test_data, desc="Generating")):
    worker_id = sample['worker_id']  # FLICKR 使用 worker_id
    history_items = sample['history_items_info']
    target_item = sample['target_item_info']
    
    # 构造历史文本
    history_captions = [
        f"{k+1}. {item.get('caption', '')}"
        for k, item in enumerate(history_items)
        if item.get('caption')
    ]
    history_text = " ".join(history_captions)
    
    # 获取 worker style（仅用于三路融合，不传给 LLaMA）
    worker_style = get_worker_style_text(worker_id)

    # 构造 prompt（不包含 style）
    prompt_text = prompt_preprocess(history_text)
    
    # Tokenize
    product_token = llama_tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()
    token_len = len(product_token)
    product_token += [llama_tokenizer.pad_token_id] * (args.max_txt_len - len(product_token))
    input_ids = torch.tensor(product_token).unsqueeze(0).to(args.device)
    token_len_tensor = torch.tensor([token_len]).to(args.device)
    
    # === 三路条件编码 ===
    # 1. Target caption embedding
    target_caption = target_item.get('caption', '')
    if target_caption:
        target_emb = sd_pipeline.textEncode(target_caption, num_tokens=40)
    else:
        target_emb = torch.zeros(1, 40, 768, device=args.device, dtype=args.weight_dtype)
    
    # 2. LLaMA image embedding
    with torch.no_grad():
        image_emb = infer_model.forward(llama_model, input_ids, token_len_tensor)
    
    # 3. Worker style embedding（从 FLICKR_styles.json 加载）
    style_emb = sd_pipeline.textEncode(worker_style, num_tokens=35)
    
    # 三路融合
    combined_emb = torch.cat([
        target_emb * args.weight_target,
        image_emb * args.weight_image,
        style_emb * args.weight_preference
    ], dim=1)
    
    # 生成图像
    generated_images = []
    for img_idx in range(args.num_images_per_sample):
        gen = sd_pipeline.generate(
            combined_emb,
            negative_prompt=args.neg_prompt,
            generator=[torch.manual_seed(args.seed + idx * 100 + img_idx)],
            show_processbar=False
        )
        generated_images.append(gen[0])
    
    # 保存输出
    sample_dir = os.path.join(args.output_dir, f"sample_{idx:04d}")
    os.makedirs(sample_dir, exist_ok=True)
    
    for img_idx, img in enumerate(generated_images):
        Image.fromarray(img).save(os.path.join(sample_dir, f"gen_{img_idx}.jpg"))
    
    save_image_grid(generated_images, os.path.join(sample_dir, "grid.jpg"))
    
    history_imgs = [_resize_rgb(item.get('image_path', ''), args.image_size) for item in history_items]
    target_img = _resize_rgb(target_item.get('image_path', ''), args.image_size)
    save_image_grid(history_imgs + [target_img], os.path.join(sample_dir, "references.jpg"))
    
    # 保存元信息（包含 worker 风格）
    meta = {
        'worker_id': worker_id,
        'worker_style': worker_style,
        'history_captions': [item.get('caption', '') for item in history_items],
        'target_caption': target_caption,
        'conditioning': {
            'target_weight': args.weight_target,
            'image_weight': args.weight_image,
            'preference_weight': args.weight_preference
        }
    }
    with open(os.path.join(sample_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print("Evaluation completed!")
print(f"Results saved to: {args.output_dir}")
print("=" * 80)