import torch
import math
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from copy import deepcopy

from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from accelerate.utils import ProjectConfiguration, set_seed

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    class Args:
        validation_steps = 100
        save_steps = 1000
        train_batch_size = 6
        vaild_batch_size = 1
        gradient_accumulation_steps = 4

        vaild_image_num = 1
        num_image_prompt = 2
        num_prefix_prompt = 2

        # 使用 epoch 而不是 max_train_steps
        num_train_epochs = 3
        learning_rate = 5e-6
        scale_lr = False
        lr_scheduler = 'linear'
        lr_warmup_steps = 0
        lr_num_cycles = 1
        adam_beta1 = 0.9
        adam_beta2 = 0.999
        adam_weight_decay = 1e-2
        adam_epsilon = 1e-06
        weight_dtype = torch.bfloat16

        resume_from = ""
        image_size = 512
        model_name = "flickr_aesthetic"
        output_dir = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/FLICKR/train_result_1/logs/{}".format(model_name)
        mixed_precision = "bf16"
        allow_tf32 = True
        report_to = "tensorboard"
        logging_dir = "logs"
        dataloader_num_workers = 4
    return Args()

args = parse_args()
set_seed(42)

import torch.backends.cuda as cuda
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ====== FLICKR 数据路径 ======
data_path = '/data-nfs/gpu1-1/ud202581869/Personalized_Generation/FLICKR/processed_dataset'
train_json_path = os.path.join(data_path, 'train.json')
val_json_path = os.path.join(data_path, 'val.json')

print("Loading training data...")
with open(train_json_path, 'r', encoding='utf-8') as f:
    train_data = json.load(f)
print(f"Loaded {len(train_data)} training samples")

print("Loading validation data...")
with open(val_json_path, 'r', encoding='utf-8') as f:
    valid_data = json.load(f)
print(f"Loaded {len(valid_data)} validation samples")

def _resize_rgb(img_path, size):
    """从路径读取并调整图像大小"""
    if not os.path.exists(img_path):
        return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    try:
        im = Image.open(img_path).convert('RGB')
        im = im.resize((size, size), Image.BICUBIC)
        arr = np.array(im, dtype=np.uint8)
        return np.ascontiguousarray(arr)
    except:
        return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

# ====== LLaMA ======
from transformers import LlamaForCausalLM, LlamaTokenizer

llama_tokenizer = LlamaTokenizer.from_pretrained("/data-nfs/gpu1-1/ud202581869/Personalized_Generation/Llama2_7b")
llama_tokenizer.pad_token = llama_tokenizer.eos_token

llama_model = LlamaForCausalLM.from_pretrained(
    "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/Llama2_7b",
    torch_dtype=args.weight_dtype,
    low_cpu_mem_usage=True,
    device_map="auto",
)
llama_model.requires_grad_(False)

# ====== Accelerator ======
logger = get_logger(__name__)
logging_dir = os.path.join(args.output_dir, args.logging_dir)
accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
)
llama_model = accelerator.prepare(llama_model)

# ====== Stable Diffusion Pipeline ======
from myCustomPipeline import SDPipeline

sd_pipeline = SDPipeline(args.weight_dtype, model_id='/data-nfs/gpu1-1/ud202581869/Personalized_Generation/stable-diffusion-v1-5')
torch.cuda.empty_cache()
sd_pipeline = accelerator.prepare(sd_pipeline)

def prompt_preprocess(history_captions):
    """构造 LLaMA prompt（不包含 style）"""
    prompt = (
        "### Human: A person rated the following images highly: \"<Images/>\". Describe their visual taste. ###Assistant: "
    )
    prompt = prompt.replace('<Images/>', history_captions)
    return prompt

# ====== Dataset ======
class FLICKRDataset(Dataset):
    def __init__(self, data, tokenizer, repeats=1, max_len=600, mode='train'):
        self.tokenizer = tokenizer
        self.mode = mode
        self.data = data
        self.max_len = max_len
        self._length = len(data) * repeats

        # 预计算所有唯一item的嵌入
        self.item_emb_dict = {}
        self.item_ids_dict = {}
        
        # 收集所有唯一的item（包括历史和target）
        unique_items = {}
        for sample in data:
            for item_info in sample['history_items_info']:
                item_id = item_info['item_id']
                if item_id not in unique_items and item_info.get('caption'):
                    unique_items[item_id] = item_info['caption']
            
            target_info = sample['target_item_info']
            item_id = target_info['item_id']
            if item_id not in unique_items and target_info.get('caption'):
                unique_items[item_id] = target_info['caption']

        print(f"[{mode}] Computing embeddings for {len(unique_items)} unique items...")
        items_list = list(unique_items.items())
        bs = 512
        for t in tqdm(range(0, len(items_list), bs), desc=f"[{mode}] Encoding"):
            batch_items = items_list[t:t+bs]
            batch_ids = [item_id for item_id, _ in batch_items]
            batch_captions = [caption for _, caption in batch_items]
            
            tokens_list = []
            for cap in batch_captions:
                tokens = sd_pipeline.textEncode(cap, num_tokens=75, return_tokens=True).detach()[0]
                tokens_list.append(tokens)
            
            embs = sd_pipeline.textEncode(tokens=torch.stack(tokens_list, dim=0))
            for i, item_id in enumerate(batch_ids):
                self.item_ids_dict[item_id] = tokens_list[i].cpu()
                self.item_emb_dict[item_id] = embs[i].detach().cpu()

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        i = i % len(self.data)
        sample = self.data[i]
        
        worker_id = sample['worker_id']
        history_items = sample['history_items_info']
        target_item = sample['target_item_info']
        worker_style = sample.get('worker_style', '')
        
        
        # 构造历史文本
        history_captions = [
            f"{k+1}. {item['caption']}"
            for k, item in enumerate(history_items)
            if item.get('caption')
        ]
        history_text = " ".join(history_captions)
        prompt_text = prompt_preprocess(history_text) 
        product_token = self.tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()

        example = {}
        example['token_len'] = len(product_token)
        assert self.max_len >= len(product_token) + args.num_image_prompt, \
            f"len:{len(product_token)} max_len:{self.max_len}"
        product_token += [llama_tokenizer.pad_token_id] * (self.max_len - len(product_token))
        example['input_ids'] = torch.tensor(product_token)

        # 正样本：target item
        target_id = target_item['item_id']
        example['keywords_ids'] = self.item_ids_dict.get(target_id, torch.zeros(77, dtype=torch.long))
        example['keywords_emb'] = self.item_emb_dict.get(target_id, torch.zeros(77, 768))
        example['pixel_values'] = _resize_rgb(target_item.get('image_path', ''), args.image_size)

        # 负样本：随机选一个其他item
        nega_id = np.random.choice(list(self.item_emb_dict.keys()))
        example['nega_keywords_ids'] = self.item_ids_dict[nega_id]
        example['nega_keywords_emb'] = self.item_emb_dict[nega_id]
        
        # 找负样本图片
        nega_img_path = None
        for s in self.data:
            if s['target_item_info']['item_id'] == nega_id:
                nega_img_path = s['target_item_info'].get('image_path')
                break
        if not nega_img_path:
            # 从历史中找
            for s in self.data:
                for h in s['history_items_info']:
                    if h['item_id'] == nega_id:
                        nega_img_path = h.get('image_path')
                        break
                if nega_img_path:
                    break
        
        example['nega_pixel_values'] = _resize_rgb(nega_img_path or '', args.image_size)

        return example

# ====== Dataloaders ======
# 不使用 repeats
train_dataset = FLICKRDataset(train_data, tokenizer=llama_tokenizer, mode='train', repeats=1)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
)
train_dataloader = accelerator.prepare(train_dataloader)

vaild_dataset = FLICKRDataset(valid_data, tokenizer=llama_tokenizer, mode='val', repeats=1)
vaild_dataloader = torch.utils.data.DataLoader(
    vaild_dataset, batch_size=args.vaild_batch_size, shuffle=False, num_workers=args.dataloader_num_workers
)
vaild_dataloader = accelerator.prepare(vaild_dataloader)

# ====== 前缀编码 + InferenceModel ======
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

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values

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
        attention_mask = torch.concat([torch.ones((bsz, self.num_prefix_prompt), device=attention_mask.device), attention_mask], dim=1)

        num_head = llama_model.model.layers[0].self_attn.num_heads
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(token.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(bsz, self.num_prefix_prompt, self.layer_num, 2, num_head, -1)
        past_key_values = past_key_values.permute(2, 3, 0, 4, 1, 5)

        outputs = llama_model.model.forward(
            inputs_embeds=emb,
            output_hidden_states=True,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        encoder_hidden_states = []
        for i in range(bsz):
            l = token_len[i].item()
            encoder_hidden_states.append(outputs.last_hidden_state[i, l:l+self.num_image_prompt])
        encoder_hidden_states = self.mapping_layer(torch.stack(encoder_hidden_states))
        return encoder_hidden_states

model = InferenceModel(
    layer_num=len(llama_model.model.layers),
    num_image_prompt=args.num_image_prompt,
    num_prefix_prompt=args.num_prefix_prompt,
    emb_dim=4096,
    sd_hidden_state_dim=768
)
model = accelerator.prepare(model)

# ====== 优化器、调度器 ======
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)

# 基于 epoch 计算 total training steps
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    num_cycles=args.lr_num_cycles * args.gradient_accumulation_steps,
)

optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

print(f"\n{'='*80}")
print(f"Training Configuration:")
print(f"  Number of training samples: {len(train_dataset)}")
print(f"  Number of epochs: {args.num_train_epochs}")
print(f"  Steps per epoch: {num_update_steps_per_epoch}")
print(f"  Total training steps: {max_train_steps}")
print(f"{'='*80}\n")

if accelerator.is_main_process:
    from datetime import datetime
    accelerator.init_trackers(
        "{}_{}_p{}_i{}".format(args.model_name, datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), args.num_prefix_prompt, args.num_image_prompt),
        config=vars(args)
    )

total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataset)}")
logger.info(f"  Num Epochs = {args.num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {max_train_steps}")

global_step = 0
first_epoch = 0

# ====== 恢复训练（可选） ======
if getattr(args, "resume_from", ""):
    resume_dir = args.resume_from
    accelerator.print(f"[RESUME] loading state from {resume_dir}")
    accelerator.load_state(resume_dir)
    import re as _re
    m = _re.search(r"epoch(\d+)", resume_dir)
    if m:
        first_epoch = int(m.group(1)) + 1
        accelerator.print(f"[RESUME] resuming from epoch {first_epoch}")

# ====== 评估可视化 ======
def log_validation(model, llama_model, sd_pipeline, global_step, batch, with_his_emb=True, with_keyword=True, name=''):
    torch.cuda.empty_cache()
    with torch.no_grad():
        keywords_emb = batch['keywords_emb']
        if with_his_emb:
            image_emb = model.forward(llama_model, batch['input_ids'], batch['token_len'])
            if with_keyword:
                image_emb = torch.concat([keywords_emb, image_emb], dim=1)
        else:
            image_emb = keywords_emb

        img_bsz = 1
        gen_images = []
        for t in range(args.vaild_image_num):
            gen_images.append(sd_pipeline.generate(
                image_emb.repeat(img_bsz, 1, 1),
                negative_prompt='lowres, text, error, cropped, worst quality, low quality',
                generator=[torch.manual_seed(i+t*img_bsz) for i in range(img_bsz)],
                show_processbar=False
            ))

        gen_images = np.concatenate(gen_images, axis=0)
        for tracker in accelerator.trackers:
            tracker.writer.add_images("validation_{}".format(name), gen_images, global_step, dataformats="NHWC")
        return gen_images

# ====== 训练 ======
progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
progress_bar.set_description("Steps")

def calcLoss(image_emb, keywords_emb, pixel_values):
    emb_loss = 0
    keywords_emb = keywords_emb.to(image_emb.device)
    image_emb = torch.concat([keywords_emb, image_emb], dim=1)
    pixel_values = (pixel_values.to(args.weight_dtype) / 127.5 - 1).permute(0, 3, 1, 2)
    pixel_values = torch.nn.functional.interpolate(pixel_values, (args.image_size, args.image_size), mode='bilinear')
    image_loss = sd_pipeline.forward(image_emb, pixel_values)
    loss = image_loss + emb_loss
    return loss, image_loss, emb_loss

for param in model.parameters():
    param.requires_grad = True

# 基于 epoch 的训练循环
for epoch in range(first_epoch, args.num_train_epochs):
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        with accelerator.accumulate(model):
            image_emb = model.forward(llama_model, batch['input_ids'], batch['token_len'])
            loss, image_loss, emb_loss = calcLoss(image_emb, batch['keywords_emb'], batch['pixel_values'])
            nega_loss, nega_image_loss, nega_emb_loss = calcLoss(image_emb, batch['nega_keywords_emb'], batch['nega_pixel_values'])
            final_loss = loss - nega_loss * 0.5
            accelerator.backward(final_loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            # 按步数保存
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"save-steps-{global_step}.pth")
                torch.save(model.state_dict(), save_path)

            if global_step % args.validation_steps == 1:
                for vstep, vbatch in enumerate(vaild_dataloader):
                    if global_step == 1:
                        his_text = llama_tokenizer.decode(vbatch['input_ids'][0])
                        for tracker in accelerator.trackers:
                            tracker.writer.add_text(
                                "validation_{}".format(vstep),
                                "### History:\n{}".format(his_text)
                            )
                            log_validation(model, llama_model, sd_pipeline, global_step, vbatch, False, name='{}_only_kw'.format(vstep))
                    _ = log_validation(model, llama_model, sd_pipeline, global_step, vbatch, name=str(vstep))

        logs = {
            'epoch': epoch,
            'loss': loss.detach().item(),
            'image_loss': image_loss.detach().item(),
            'lr': lr_scheduler.get_last_lr()[0],
            'sync_gradients': accelerator.sync_gradients
        }
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
    
    # Epoch 结束后保存
    save_path = os.path.join(args.output_dir, f"model-epoch{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n{'='*80}")
    print(f"Completed Epoch {epoch+1}/{args.num_train_epochs}")
    print(f"Saved: {save_path}")
    print(f"{'='*80}\n")

accelerator.wait_for_everyone()
accelerator.end_training()