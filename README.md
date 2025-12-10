# Personalized Multimodal Generation Benchmark

A comprehensive benchmark for evaluating personalized image generation methods across multiple datasets and metrics.

## 📋 Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Baseline Methods](#baseline-methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Benchmark Results](#benchmark-results)
- [Data Format](#data-format)
- [Getting Started](#getting-started)
- [Citation](#citation)

---

## 🎯 Overview

This benchmark evaluates personalized image generation methods that learn and adapt to individual user preferences. The benchmark includes three diverse datasets (FLICKR-AES, POG, SER30K) and three state-of-the-art baseline methods (Textual Inversion, IP-Adapter, PMG), evaluated across eight comprehensive metrics.

---

## 📊 Datasets

### 1. FLICKR-AES Dataset

**Source:** [FLICKR-AES](https://github.com/alanspike/personalizedImageAesthetics)

**Description:** A curated collection of Creative Commons-licensed photos from FLICKR, rated for aesthetic quality.

**Statistics:**
- **Images:** 40,988 photos
- **Workers:** 210 annotators
- **Total Ratings:** 193,208 annotations
- **Rating Scale:** 1-5 (aesthetic quality)
- **Annotation Method:** Amazon Mechanical Turk (5 workers per image)

**Use Case:** Personalized aesthetic preference learning

**Data Files:**
- `FLICKR-AES_image_labeled_by_each_worker.csv` - Individual worker ratings
- `FLICKR-AES_image_score.txt` - Aggregated image scores
- `FLICKR_captions.json` - Image captions
- `FLICKR_styles.json` - Worker style preferences (210 entries)
- `40K/` - Image directory

### 2. POG (Polyvore Outfit Generation) Dataset

**Source:** [POG](https://github.com/wenyuer/POG) 

**Description:** A multimodal fashion dataset containing outfit combinations and user interactions.

**Statistics:**
- **Users:** 2,000 selected users
- **Items:** 16,100 fashion items
- **Domain:** Fashion clothing and accessories

**Use Case:** Personalized fashion recommendation and generation

**Data Files:**
- `user_data.txt` - User interaction sequences
- `outfit_data.txt` - Outfit combinations
- `item_data.txt` - Item metadata
- `captions_sampled.json` - Item descriptions
- `user_styles.json` - User style preferences (2,000 entries)
- `images_sampled/` - Product images

### 3. SER30K Dataset

**Source:** [SER30K](https://github.com/LizhenWangXDU/SER30K)

**Description:** A large-scale sticker and emoji dataset with emotion labels and theme categorization.

**Statistics:**
- **Stickers:** 30,000+ stickers
- **Categories:** Multiple emotion and theme categories
- **Annotation:** Emotion labels per sticker

**Use Case:** Personalized sticker and emoji generation

**Data Files:**
- `Images/` - Sticker image collections organized by theme
- `Annotations/` - Emotion and category labels
- `ser30k_captions.json` - Sticker descriptions
- `user_preferences.json` - User preference keywords
- `id_map.csv` - ID mapping information

---

## 🔧 Baseline Methods

### 1. Textual Inversion (TI)

**Paper:** [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618)

**Description:** Textual Inversion introduces learnable word embeddings to represent user preferences. These embeddings are combined with textual instructions to guide the text-to-image generation process in Diffusion Models.

**Key Features:**
- Learns a single word embedding per concept/user
- Integrates seamlessly with pre-trained text-to-image models
- Lightweight and efficient approach

**Implementation:** `method/textual_inversion/`

### 2. IP-Adapter

**Paper:** [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721)

**Description:** IP-Adapter is an effective and lightweight adapter (22M parameters) that achieves image prompt capability for pre-trained text-to-image diffusion models. It can be generalized to custom models and supports multimodal (image + text) generation.

**Key Features:**
- Only 22M parameters
- Achieves comparable or better performance than fine-tuned models
- Supports controllable generation with existing tools
- Enables multimodal image generation

**Implementation:** `method/ip-adapter/`

### 3. PMG (Preference-based Multimodal Generation)

**Paper:** [PMG: Personalized Multimodal Generation with Large Language Models](https://arxiv.org/abs/2404.08677)

**Description:** PMG transforms user-interacted and reference images into textual descriptions, using pre-trained Large Language Models (LLMs) to extract user preferences through keywords and implicit embeddings to condition the image generator.

**Key Features:**
- Leverages LLMs for preference understanding
- Combines explicit keywords and implicit embeddings
- Three-way conditioning mechanism
- Handles sequential user interactions

**Implementation:** `method/PMG/`

---

## 📈 Evaluation Metrics

### 1. LPIPS (Learned Perceptual Image Patch Similarity)

**Range:** [0, 1] (lower is better for similarity)

**Description:** Measures perceptual similarity between generated and reference images using deep features from VGG networks.

**Variants:**
- **LPIPS (vs Target):** Perceptual distance to the target image
- **LPIPS (vs History Avg):** Average perceptual distance to historical user-interacted images

**Interpretation:** Lower values indicate better perceptual similarity.

### 2. SSIM (Structural Similarity Index Measure)

**Range:** [-1, 1] (higher is better)

**Description:** Assesses structural similarity between images based on luminance, contrast, and structure.

**Variants:**
- **SSIM (vs Target):** Structural similarity to the target image
- **SSIM (vs History Avg):** Average structural similarity to historical images

**Interpretation:** Higher values indicate better structural preservation.

### 3. CPS (CLIP Preference Similarity)

**Range:** [-1, 1] (higher is better)

**Description:** Measures alignment between generated images and user preference text descriptions using CLIP embeddings.

**Calculation:** Cosine similarity between CLIP image features and user preference text features.

**Interpretation:** Higher scores indicate better alignment with user preferences.

### 4. CPIS (CLIP Preference Image Similarity)

**Range:** [-1, 1] (higher is better)

**Description:** Measures visual similarity between generated images and user's historical images in CLIP feature space.

**Variant:**
- **CPIS (vs History Avg):** Average CLIP similarity to historical user images

**Interpretation:** Higher values indicate better consistency with user's visual style.

### 5. HPSv2 (Human Preference Score v2)

**Range:** [0, 1] (higher is better)

**Description:** Predicts human aesthetic preferences for text-to-image generation using a trained alignment model.

**Calculation:** Evaluates alignment between generated images and their text prompts based on human preference data.

**Interpretation:** Higher scores indicate better alignment with human aesthetic preferences.

### 6. LAION Aesthetic Score

**Range:** Typically [0, 10] (higher is better)

**Description:** Predicts aesthetic quality using a predictor trained on LAION-Aesthetics dataset with CLIP features.

**Calculation:** Linear projection of CLIP image features to aesthetic scores.

**Interpretation:** Higher scores indicate better aesthetic quality.

### 7. Verifier Score (FLICKR only)

**Range:** [0, 1] (higher is better)

**Description:** A learned verifier model that predicts whether an image matches a specific worker's aesthetic preferences.

**Calculation:** Binary classification score from a fine-tuned verifier network.

**Interpretation:** Higher scores indicate better personalization to individual worker preferences.

---

## 🏆 Benchmark Results

**Notes:**
- ↓ indicates lower is better, ↑ indicates higher is better
- **Bold** values indicate best performance per dataset
- Verifier Score is only applicable to FLICKR dataset
- All values are reported as mean across test samples

### Detailed Results

#### SER Dataset

<details>
<summary>Click to expand</summary>

| Method | LPIPS↓ (Target) | LPIPS↓ (History) | SSIM↑ (Target) | SSIM↑ (History) | CPS↑ | CPIS↑ (History) | HPSv2↑ | Aesthetic↑ |
|--------|----------------|-----------------|---------------|----------------|-----|----------------|-------|-----------|
| Textual Inversion | 0.7782±0.0635 | 0.8025±0.0584 | 0.2707±0.1194 | 0.2591±0.0977 | 0.1953±0.0373 | 0.5467±0.0845 | 0.2376±0.0392 | 5.8417±0.8150 |
| IP-Adapter | 0.7269±0.0847 | 0.6971±0.0842 | 0.2654±0.1098 | 0.2838±0.1271 | 0.2580±0.0414 | 0.7480±0.1137 | 0.1620±0.0530 | 5.1885±0.9788 |
| PMG | 0.7841±0.0600 | 0.7878±0.0526 | 0.2690±0.1093 | 0.2785±0.0990 | 0.2357±0.0413 | 0.5705±0.1073 | 0.2009±0.0389 | 7.1608±0.9049 |

</details>

#### POG Dataset

<details>
<summary>Click to expand</summary>

| Method | LPIPS↓ (Target) | LPIPS↓ (History) | SSIM↑ (Target) | SSIM↑ (History) | CPS↑ | CPIS↑ (History) | HPSv2↑ | Aesthetic↑ |
|--------|----------------|-----------------|---------------|----------------|-----|----------------|-------|-----------|
| Textual Inversion | 0.7167±0.0502 | 0.7407±0.0360 | 0.2533±0.1243 | 0.2500±0.0778 | 0.2247±0.0418 | 0.6619±0.0703 | 0.2409±0.0310 | 5.7650±0.7018 |
| IP-Adapter | 0.6859±0.0636 | 0.6863±0.0515 | 0.2452±0.1225 | 0.2569±0.1004 | 0.2401±0.0397 | 0.7477±0.0636 | 0.1762±0.0530 | 5.0359±0.8203 |
| PMG | 0.7034±0.0622 | 0.7224±0.0528 | 0.2353±0.1282 | 0.2247±0.1039 | 0.2548±0.0421 | 0.7023±0.0831 | 0.1897±0.0486 | 7.7765±0.3973 |

</details>

#### FLICKR Dataset

<details>
<summary>Click to expand</summary>

| Method | LPIPS↓ (Target) | LPIPS↓ (History) | SSIM↑ (Target) | SSIM↑ (History) | CPS↑ | CPIS↑ (History) | HPSv2↑ | Aesthetic↑ | Verifier↑ |
|--------|----------------|-----------------|---------------|----------------|-----|----------------|-------|-----------|----------|
| Textual Inversion | 0.8019±0.0607 | 0.8086±0.0352 | 0.2957±0.1120 | 0.2949±0.0772 | 0.1536±0.0461 | 0.4735±0.0696 | 0.1925±0.0475 | 5.1527±1.1666 | 0.4475±0.2839 |
| IP-Adapter | 0.7930±0.0675 | 0.7624±0.0497 | 0.3223±0.0915 | 0.3420±0.1081 | 0.1835±0.0437 | 0.6546±0.0867 | 0.1294±0.0480 | 5.4714±0.6464 | 0.7108±0.2455 |
| PMG | 0.7487±0.0613 | 0.7955±0.0456 | 0.3102±0.1190 | 0.2889±0.0804 | 0.1907±0.0468 | 0.5002±0.0639 | 0.2255±0.0399 | 7.4664±0.5896 | 0.6475±0.2839 |

</details>

---

## 📁 Data Format

### Processed Dataset Structure

All datasets follow a unified JSON format for train/validation/test splits:


```json
  {
    "user_id": "unique_user_identifier",
    "worker_id": "unique_worker_identifier",  // FLICKR only
    "history_item_ids": ["item1", "item2", "item3"],
    "history_items_info": [
      {
        "item_id": "item1",
        "caption": "A vibrant red crossbody bag...",
        "image_path": "/path/to/image1.jpg",
        "score": 4,  // FLICKR only
        "aesthetic_score": 0.75  // FLICKR only
      }
    ],
    "target_item_id": "target_item",
    "target_item_info": {
      "item_id": "target_item",
      "caption": "A stylish blue jacket...",
      "image_path": "/path/to/target.jpg"
    },
    "user_style": "Keywords describing user preference",  // POG, SER
    "worker_style": "Keywords describing worker preference",  // FLICKR
    "num_interactions": 3,
    "window_position": 154,  // POG, FLICKR
    "total_sequence_length": 783  // POG, FLICKR
  }
```
### User Preference Files

#### FLICKR: `FLICKR_styles.json`

```json
  {
    "worker": "WORKER_ID",
    "style": "Nature, landscapes, vibrant colors, serene, dramatic..."
  }
```
#### POG: `user_styles.json`

```json
  {
    "user": "USER_HASH",
    "style": "Elegant, vibrant, classic, sophisticated..."
  }
```
#### SER: `user_preferences.json`

```json
{
  "topic-name": {
    "topic": "topic-name",
    "num_history_items": 25,
    "keywords": ["keyword1", "keyword2", ...],
    "sample_captions": ["caption1", "caption2", ...]
  }
}
```
## 🚀 Getting Started

### Environment Setup

# Create conda environment
conda env create -n pmg-bench python==3.10

conda activate pmg_bench

# install requirements
pip install -r requirements.txt

#### Textual Inversion
cd method/textual_inversion
##### Follow instructions in evaluate_*.py

#### IP-Adapter
cd method/ip-adapter
##### Follow instructions in README

#### PMG
cd method/PMG
##### Train on FLICKR
python FLICKR_PMG_TRAIN.py

##### Train on POG
python POG_PMG_TRAIN.py

##### Train on SER
python SER_PMG_TRAIN.py

### Evaluation

python evaluation.py --dataset FLICKR --device cuda
python evaluation.py --dataset POG --device cuda
python evaluation.py --dataset SER --device cuda

## 📊 Key Findings

1. **PMG consistently achieves the highest aesthetic scores** across all datasets, significantly outperforming baselines.

2. **IP-Adapter shows strong performance on history-based metrics** (LPIPS and CPIS vs History), indicating good style consistency.

3. **Textual Inversion demonstrates balanced performance** across different metrics but with lower aesthetic quality.

4. **Dataset characteristics matter**: 
   - FLICKR (aesthetic photos): Higher SSIM scores across methods
   - POG (fashion): Best CPIS performance from IP-Adapter
   - SER (stickers): Most varied metric distributions

---

## 📝 License

This benchmark is released under the MIT License. Individual datasets may have their own licenses - please refer to their respective sources.

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues or pull requests to improve the benchmark.

---

## 🙏 Acknowledgments

We thank the creators of FLICKR-AES, POG, and SER30K datasets, as well as the developers of the baseline methods for making their work publicly available.
