# Personalized Image Generation Project

This project contains code for personalized image generation, supporting FLICKR-AES, POG, and SER datasets.

## Configuration

All absolute paths in the project have been replaced with placeholders. Before use, please replace all placeholders with actual paths.

### Placeholder List

#### Dataset Paths
- `{FLICKR_AES_BASE_PATH}` - Base path for FLICKR-AES dataset
  - Contains subdirectories: `processed_dataset/`, `40K/`, `FLICKR_captions.json`, `FLICKR_styles.json`, etc.

- `{POG_BASE_PATH}` - Base path for POG dataset
  - Contains subdirectories: `processed_dataset/`, `images_sampled/`, `POG_captions_sampled.json`, etc.

- `{SER_DATASET_BASE_PATH}` - Base path for SER dataset
  - Contains subdirectories: `processed/`, `processed_masked/`, `Images/`, `Annotations/`, etc.

#### Model Paths
- `{SD15_MODEL_PATH}` - Stable Diffusion 1.5 model path

- `{IP_ADAPTER_PATH}` - IP-Adapter model path
  - Should contain `models/` subdirectory and weight files (e.g., `ip-adapter_sd15.bin`)

- `{LLM_MODEL_PATH}` - LLM model path for caption masking
  - Used by `mask_*_captions_with_llm.py` scripts

- `{CLIP_VIT_H14_MODEL_PATH}` - CLIP ViT-H-14 model path (optional)
  - Used for feature extraction in some evaluation scripts

- `{LAION_AESTHETIC_MODEL_PATH}` - LAION Aesthetic predictor model path
  - Used for calculating LAION aesthetic score

### How to Replace Placeholders

#### Method 1: Batch Replacement Using sed Command (Recommended)

```bash
# Replace placeholders in all Python files
find . -name "*.py" -type f -exec sed -i 's|{FLICKR_AES_BASE_PATH}|/your/actual/path/FLICKR-AES|g' {} +
find . -name "*.py" -type f -exec sed -i 's|{POG_BASE_PATH}|/your/actual/path/POG|g' {} +
find . -name "*.py" -type f -exec sed -i 's|{SER_DATASET_BASE_PATH}|/your/actual/path/SER_Dataset|g' {} +
find . -name "*.py" -type f -exec sed -i 's|{SD15_MODEL_PATH}|/your/actual/path/stable-diffusion-v1-5|g' {} +
find . -name "*.py" -type f -exec sed -i 's|{IP_ADAPTER_PATH}|/your/actual/path/IP-Adapter|g' {} +
find . -name "*.py" -type f -exec sed -i 's|{LLM_MODEL_PATH}|/your/actual/path/Qwen3-4B-Instruct-2507|g' {} +
find . -name "*.py" -type f -exec sed -i 's|{CLIP_VIT_H14_MODEL_PATH}|/your/actual/path/CLIP-ViT-H-14-laion2B-s32B-b79K|g' {} +
find . -name "*.py" -type f -exec sed -i 's|{LAION_AESTHETIC_MODEL_PATH}|/your/actual/path/aesthetic-predictor/sa_0_4_vit_b_32_linear.pth|g' {} +
```

#### Method 2: Manual File Editing

Use a text editor or IDE's find-and-replace functionality to replace each placeholder one by one.

## Data Preprocessing

### 1. Prepare Datasets

```bash
# Prepare FLICKR-AES dataset
python data_process/prepare_flickr_aes_dataset.py

# Prepare POG dataset
python data_process/prepare_pog_dataset_split.py

# Prepare SER dataset
python data_process/prepare_ser_dataset.py
```

### 2. Extract Image Embeddings (for FLICKR-AES)

```bash
python data_process/extract_image_embeddings.py
```

### 3. Train Verifier (for FLICKR-AES)

```bash
python data_process/train_verifier.py
```

### 4. Caption Masking

```bash
# FLICKR-AES
python textual_inversion/mask_flickr_captions_with_llm.py

# POG
python textual_inversion/mask_pog_captions_with_llm.py

# SER
python textual_inversion/mask_captions_with_llm.py
```

**Note**: These scripts require correct paths to be set in the code (replace placeholders).

## Running the Project

Each dataset supports two methods: **IP-Adapter** and **Textual Inversion**.

### FLICKR-AES Dataset

#### Method 1: IP-Adapter

```bash
python ip-adater/personalized_generation_flickr.py \
    --test_json {FLICKR_AES_BASE_PATH}/processed_dataset/test.json \
    --captions_path {FLICKR_AES_BASE_PATH}/FLICKR_captions.json \
    --image_dir {FLICKR_AES_BASE_PATH}/40K \
    --output_dir {FLICKR_AES_BASE_PATH}/ip_adapter_generated \
    --sd_model_path {SD15_MODEL_PATH} \
    --ip_adapter_path {IP_ADAPTER_PATH} \
    --ip_adapter_weight ip-adapter_sd15.bin \
    --verifier_model_path {FLICKR_AES_BASE_PATH}/verifier_checkpoints/best_model.pth \
    --verifier_user_map_path {FLICKR_AES_BASE_PATH}/verifier_checkpoints/user_map.json \
    --styles_path {FLICKR_AES_BASE_PATH}/FLICKR_styles.json \
    --device cuda \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --ip_adapter_scale 0.6 \
    --max_samples 50 \
    --start_idx 0 \
    --seed 42
```

#### Method 2: Textual Inversion

**Step 1: Train Textual Inversion**

```bash
python textual_inversion/flickr_aes_textual_inversion_sd15.py
```

**Step 2: Evaluate**

```bash
python textual_inversion/evaluate_flickr_aes.py \
    --test_json {FLICKR_AES_BASE_PATH}/processed_dataset/test.json \
    --embedding_path {FLICKR_AES_BASE_PATH}/textual_inversion_sd15/learned_embeds.bin \
    --sd15_path {SD15_MODEL_PATH} \
    --output_dir {FLICKR_AES_BASE_PATH}/evaluation_results \
    --generated_images_dir {FLICKR_AES_BASE_PATH}/evaluation_generated_images \
    --images_dir {FLICKR_AES_BASE_PATH}/40K \
    --captions_json {FLICKR_AES_BASE_PATH}/FLICKR_captions_masked.json \
    --verifier_model_path {FLICKR_AES_BASE_PATH}/verifier_checkpoints/best_model.pth \
    --verifier_user_map_path {FLICKR_AES_BASE_PATH}/verifier_checkpoints/user_map.json \
    --styles_path {FLICKR_AES_BASE_PATH}/FLICKR_styles.json \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --seed 42
```

### POG Dataset

#### Method 1: IP-Adapter

```bash
python ip-adater/personalized_generation_pog.py \
    --test_json {POG_BASE_PATH}/processed_dataset/test.json \
    --output_dir {POG_BASE_PATH}/ip_adapter_generated \
    --sd_model_path {SD15_MODEL_PATH} \
    --ip_adapter_path {IP_ADAPTER_PATH} \
    --ip_adapter_weight ip-adapter_sd15.bin \
    --device cuda \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --ip_adapter_scale 0.6 \
    --seed 42
```

#### Method 2: Textual Inversion

**Step 1: Train Textual Inversion**

```bash
python textual_inversion/pog_textual_inversion_sd15.py
```

**Step 2: Evaluate**

```bash
python textual_inversion/evaluate_pog.py \
    --test_json {POG_BASE_PATH}/processed_dataset/test.json \
    --embedding_path {POG_BASE_PATH}/textual_inversion_sd15/learned_embeds.bin \
    --sd15_path {SD15_MODEL_PATH} \
    --output_dir {POG_BASE_PATH}/evaluation_results \
    --generated_images_dir {POG_BASE_PATH}/evaluation_generated_images \
    --images_dir {POG_BASE_PATH}/images_sampled \
    --captions_json {POG_BASE_PATH}/POG_captions_sampled.json \
    --masked_captions_json {POG_BASE_PATH}/POG_captions_sampled_masked.json \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --seed 42 \
    --use_masked_caption
```

### SER Dataset

#### Method 1: IP-Adapter

```bash
python ip-adater/personalized_generation.py \
    --test_json {SER_DATASET_BASE_PATH}/processed/test.json \
    --output_dir {SER_DATASET_BASE_PATH}/ip_adapter_generated \
    --sd_model_path {SD15_MODEL_PATH} \
    --ip_adapter_path {IP_ADAPTER_PATH} \
    --ip_adapter_weight ip-adapter_sd15.bin \
    --device cuda \
    --num_inference_steps 50 \
    --guidance_scale 10.0 \
    --ip_adapter_scale 0.3 \
    --seed 42
```

#### Method 2: Textual Inversion

**Step 1: Train Textual Inversion**

```bash
python textual_inversion/ser_textual_inversion_sd15.py
```

**Step 2: Evaluate**

```bash
python textual_inversion/evaluate_ser.py \
    --test_json {SER_DATASET_BASE_PATH}/processed/test.json \
    --embedding_path {SER_DATASET_BASE_PATH}/textual_inversion_sd15/learned_embeds.bin \
    --sd15_path {SD15_MODEL_PATH} \
    --output_dir {SER_DATASET_BASE_PATH}/evaluation_results \
    --generated_images_dir {SER_DATASET_BASE_PATH}/evaluation_generated_images \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --seed 42
```

## Requirements

### Python Packages

Main dependencies include:
- PyTorch
- diffusers
- transformers
- torchvision
- PIL (Pillow)
- numpy
- pandas
- lpips
- torchmetrics
- clip
- hpsv2 (optional)
- transformers (for LLM)

## Directory Structure

```
.
├── data_process/              # Data preprocessing scripts
│   ├── prepare_flickr_aes_dataset.py
│   ├── prepare_pog_dataset_split.py
│   ├── prepare_ser_dataset.py
│   ├── extract_image_embeddings.py
│   └── train_verifier.py
├── ip-adater/                # IP-Adapter related scripts
│   ├── personalized_generation_flickr.py
│   ├── personalized_generation_pog.py
│   └── personalized_generation.py
├── textual_inversion/        # Textual Inversion related scripts
│   ├── evaluate_*.py         # Evaluation scripts
│   ├── mask_*_captions_with_llm.py  # Caption masking scripts
│   └── *_textual_inversion_sd15.py  # Training scripts
└── README.md                  # This file
```

## Important Notes

1. **Path Replacement**: Before using any script, you must replace all placeholders with actual paths.

2. **Data Preprocessing**: Complete all data preprocessing steps before running generation or evaluation scripts.

3. **GPU Requirements**: Most scripts require GPU support. Ensure CUDA is available.

4. **Memory Requirements**: Some scripts (e.g., generation and evaluation) require large GPU memory (recommended 16GB+).

5. **Dependency Installation**: Ensure all Python dependencies are correctly installed.

## FAQ

### Q: How do I know which files contain placeholders?

A: Use the following command to search:
```bash
grep -r "{.*_PATH}" --include="*.py" .
```

### Q: Scripts still report file not found errors after replacing placeholders?

A: Check:
1. Whether paths are correct (pay attention to case sensitivity and slashes)
2. Whether files/directories exist
3. Whether you have access permissions

### Q: How to replace placeholders on Windows?

A: You can use PowerShell's replace functionality, or use a text editor with regex support (such as VS Code) for batch replacement.

## License

Please refer to the LICENSE file in the project root directory (if it exists).
