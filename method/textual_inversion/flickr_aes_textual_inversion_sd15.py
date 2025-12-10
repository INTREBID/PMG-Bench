import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

@dataclass
class FLICKRAESConfig:
    train_json: str = "{FLICKR_AES_BASE_PATH}/processed_dataset/train.json"
    masked_captions_file: str = "{FLICKR_AES_BASE_PATH}/FLICKR_captions_masked.json"
    original_captions_file: str = "{FLICKR_AES_BASE_PATH}/FLICKR_captions.json"
    sd15_path: str = "{SD15_MODEL_PATH}"
    target_token: str = "[V]"
    initializer_token: str = "photo"
    num_vectors: int = 8
    image_size: int = 512
    train_batch_size: int = 2
    max_train_steps: int = 3000
    lr: float = 5e-04
    output_dir: str = "{FLICKR_AES_BASE_PATH}/textual_inversion_sd15"
    log_image_dir: str = "{FLICKR_AES_BASE_PATH}/train_images"
    log_interval: int = 200


class FLICKRAESMaskedCaptionDataset(Dataset):
    def __init__(self, json_path: str, masked_captions_dict: Dict[str, str], 
                 original_captions_dict: Dict[str, str],
                 placeholder_tokens_str: str, target_token_symbol: str, image_size: int = 512):
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            self.data: List[Dict] = json.load(f)
        self.image_size = image_size
        self.placeholder_str = placeholder_tokens_str
        self.target_symbol = target_token_symbol
        self.masked_captions_dict = masked_captions_dict
        self.original_captions_dict = original_captions_dict
        
        
        self.total_items = 0
        self.items_with_masked = 0
        self.items_without_masked = 0
        for sample in self.data:
            interaction_sequence = sample.get("interaction_sequence", [])
            for item in interaction_sequence:
                self.total_items += 1
                item_id = item.get("item_id", "")
                item_id_key = item_id.replace(".jpg", "") if item_id.endswith(".jpg") else item_id
                if item_id_key in self.masked_captions_dict:
                    self.items_with_masked += 1
                else:
                    self.items_without_masked += 1
        
        print(f"Dataset loaded: {len(self.data)} samples")
        print(f"  - Total interaction items: {self.total_items}")
        print(f"  - Items with masked caption: {self.items_with_masked} ({self.items_with_masked/self.total_items*100:.1f}%)")
        print(f"  - Items without masked caption: {self.items_without_masked} ({self.items_without_masked/self.total_items*100:.1f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        max_sample_attempts = 50
        item = None
        image_path = None
        item_id = None
        train_prompt = None
        sample = None
        interaction_sequence = None
        
        for sample_attempt in range(max_sample_attempts):
            if sample_attempt > 0:
                idx = random.randint(0, len(self.data) - 1)
            
            sample = self.data[idx]
            interaction_sequence = sample.get("interaction_sequence", [])
            if not interaction_sequence:
                continue

            max_attempts = 10
            found_valid_item = False
            for attempt in range(max_attempts):
                item = random.choice(interaction_sequence)
                image_path = item.get("image_path")
                item_id = item.get("item_id", "")

                if not image_path or not os.path.exists(image_path):
                    continue

                item_id_key = item_id.replace(".jpg", "") if item_id.endswith(".jpg") else item_id
                masked_caption = self.masked_captions_dict.get(item_id_key, "")
                if not masked_caption:
                    continue

                train_prompt = masked_caption.replace(self.target_symbol, self.placeholder_str)
                
                if self.placeholder_str not in train_prompt:
                    continue
                
                found_valid_item = True
                break
            
            if found_valid_item:
                break
        
        if not found_valid_item or item is None or image_path is None or train_prompt is None or sample is None or interaction_sequence is None:
            raise ValueError(f"Failed to find valid data after {max_sample_attempts} attempts")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        image = np.array(image).astype(np.float32) / 255.0
        image = (image * 2.0) - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        target_prompt = ""
        for target_item in interaction_sequence:
            target_item_id = target_item.get("item_id", "")
            if target_item_id and target_item_id != item_id:
                target_item_id_key = target_item_id.replace(".jpg", "") if target_item_id.endswith(".jpg") else target_item_id
                masked_caption = self.masked_captions_dict.get(target_item_id_key, "")
                
                if masked_caption:
                    if self.target_symbol not in masked_caption:
                        continue
                    else:
                        target_prompt = masked_caption.replace(self.target_symbol, self.placeholder_str)
                        break
        
        

        return {
            "pixel_values": image,
            "prompt": train_prompt,  # This is the expanded prompt
            "meta": {
                "user_id": sample.get("user_id", "unknown"),
                "target_prompt": target_prompt,
                "item_id": item_id,
            },
        }


def main():
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        raise ImportError("pip install diffusers transformers accelerate")

    cfg = FLICKRAESConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_image_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load masked captions (for training)
    print(f"Loading masked captions from {cfg.masked_captions_file}...")
    if not os.path.exists(cfg.masked_captions_file):
        print(f"Warning: Masked captions file not found: {cfg.masked_captions_file}")
        print("Will use original captions instead.")
        masked_captions_dict = {}
    else:
        with open(cfg.masked_captions_file, "r", encoding="utf-8") as f:
            masked_captions_dict = json.load(f)
        print(f"Loaded {len(masked_captions_dict)} masked captions")
    
    # Load original full descriptions (for test generation)
    print(f"Loading original captions from {cfg.original_captions_file}...")
    if not os.path.exists(cfg.original_captions_file):
        print(f"Warning: Original captions file not found: {cfg.original_captions_file}")
        original_captions_dict = {}
    else:
        with open(cfg.original_captions_file, "r", encoding="utf-8") as f:
            original_captions_dict = json.load(f)
        print(f"Loaded {len(original_captions_dict)} original captions")

    print("Loading SD1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(cfg.sd15_path, safety_checker=None)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    pipe.to(device)
    text_encoder.to(dtype=torch.float32)
    vae.to(dtype=torch.float16)
    unet.to(dtype=torch.float16)

    placeholder_tokens = [f"<v_{i}>" for i in range(cfg.num_vectors)]
    placeholder_tokens_str = " ".join(placeholder_tokens)
    print(f"Training token sequence: {placeholder_tokens_str}")

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != cfg.num_vectors:
        raise ValueError(f"Failed to add tokens. Expected {cfg.num_vectors}, got {num_added_tokens}")

    text_encoder.resize_token_embeddings(len(tokenizer))

    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    print(f"Placeholder token IDs: {placeholder_token_ids}")

    initializer_token_ids = tokenizer.encode(cfg.initializer_token, add_special_tokens=False)
    if len(initializer_token_ids) == 0:
        raise ValueError(f"Initializer token '{cfg.initializer_token}' not found in tokenizer!")
    init_token_id = initializer_token_ids[0]

    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        init_vec = token_embeds[init_token_id].clone()
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = init_vec

    embeds = text_encoder.get_input_embeddings().weight
    embeds.requires_grad_(True)
    optimizer = torch.optim.AdamW([embeds], lr=cfg.lr)

    grad_mask = torch.zeros_like(embeds)
    grad_mask[placeholder_token_ids, :] = 1.0
    grad_mask = grad_mask.to(device)

    dataset = FLICKRAESMaskedCaptionDataset(
        cfg.train_json,
        masked_captions_dict=masked_captions_dict,
        original_captions_dict=original_captions_dict,
        placeholder_tokens_str=placeholder_tokens_str,
        target_token_symbol=cfg.target_token,
        image_size=cfg.image_size,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=4)
    noise_scheduler = pipe.scheduler

    print(f"Start Training (Standard Forward Pass)... Vectors: {cfg.num_vectors}, Steps: {cfg.max_train_steps}")
    global_step = 0
    progress_bar = tqdm(total=cfg.max_train_steps)

    while global_step < cfg.max_train_steps:
        for batch in dataloader:
            if global_step >= cfg.max_train_steps:
                break

            # Prepare input
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            prompts = batch["prompt"]

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

            tokenized = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(device)

            encoder_hidden_states = text_encoder(input_ids)[0].to(dtype=torch.float16)

            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                text_encoder.get_input_embeddings().weight.grad *= grad_mask

            optimizer.step()

            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix(loss=loss.item())

            if global_step % cfg.log_interval == 0:
                current_embeds = text_encoder.get_input_embeddings().weight[placeholder_token_ids].detach().cpu()

                save_path = os.path.join(cfg.output_dir, f"learned_embeds_step_{global_step}.bin")
                torch.save({cfg.target_token: current_embeds}, save_path)
                print(f"\nStep {global_step} Saving & Testing...")
                print(f"Saved embeddings to {save_path}")

                text_encoder.eval()
                with torch.no_grad():
                    test_meta = batch["meta"]
                    t_user_id = test_meta["user_id"][0] if isinstance(test_meta["user_id"], list) else test_meta["user_id"]
                    t_prompt = test_meta["target_prompt"][0] if isinstance(test_meta["target_prompt"], list) else test_meta["target_prompt"]

                    if not t_prompt or not t_prompt.strip():
                        print(f"  - user_id: {t_user_id}")
                        print(f"  - Skipping test generation (target_prompt is empty)")
                        continue
                    
                    if cfg.target_token in t_prompt:
                        print(f"  - Warning: target_prompt still contains [V], replacing...")
                        t_prompt = t_prompt.replace(cfg.target_token, placeholder_tokens_str)
                    
                    if placeholder_tokens_str not in t_prompt:
                        print(f"  - Warning: target_prompt does not contain placeholder tokens")
                        continue
                    
                    print(f"  - user_id: {t_user_id}")
                    print(f"  - Generation prompt:")
                    print(f"    {t_prompt}")

                    pipe.to(dtype=torch.float16)
                    try:
                        image = pipe(t_prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                    finally:
                        text_encoder.to(dtype=torch.float32)

                    img_path = os.path.join(cfg.log_image_dir, f"{t_user_id}_step_{global_step}.png")
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    image.save(img_path)
                    print(f"  - Saved image to: {img_path}\n")

                text_encoder.train()

    # Final Save
    final_embeds = text_encoder.get_input_embeddings().weight[placeholder_token_ids].detach().cpu()
    final_path = os.path.join(cfg.output_dir, "learned_embeds.bin")
    torch.save({cfg.target_token: final_embeds}, final_path)
    print(f"\nTraining completed! Saved final embeddings to: {final_path}")


if __name__ == "__main__":
    main()

