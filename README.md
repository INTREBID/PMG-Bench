# PMG-Bench
A unified dataset for personalized multimodal generation in recommender systems, covering movies, e-commerce fashion, and sticker scenarios.

# 🚀 Overview

**PMG-Bench** is a large-scale benchmark designed for **personalized multimodal generation** in recommendation scenarios.  
It covers three major domains:

- 🎬 **Movies** — Movie posters + user ratings (MovieLens-derived)  
- 👗 **Fashion** — Outfit and item images + captions (POG dataset)  
- 😂 **Stickers/Emojis** — SER30K sticker images + textual annotations  

To enhance cross-modal understanding, all images are **recaptioned using Qwen2.5VL-7B**, producing two semantic channels:

- **Style / Preference Caption** — representing user historical preference signals  
- **Target Caption** — representing the semantics of the target item  

PMG-Bench is built for training and evaluating models such as:

- personalized poster generation  
- personalized outfit image generation  
- personalized caption generation  
- multimodal recommendation explanation  
- user-conditioned image editing / generation  

---

# 📊 Key Statistics

| Domain | #Images | #Users | #Interactions | Captions | Notes |
|--------|---------|--------|---------------|----------|--------|
| Movie posters | 9,742 | 610 | 100,836 | style + target | Derived from MovieLens |
| Fashion outfits | ~150k | ~50k | ~800k | visual captions | Outfit & item hierarchy |
| Stickers (SER30K) | 30k | synthetic | category-based | visual captions | Meme/emoji generation |

**Total Images:** ~230,000  
**Total User–Item Interactions:** ~1,000,000  

---

# 📁 Repository Structure

```
PMG-Bench/
│
├── movieposter/                 # Personalized movie poster generation
│   ├── info.csv                 # Movie metadata
│   ├── ratings.csv              # User–movie ratings
│   ├── style_captions.pkl       # User preference / style captions
│   ├── target_captions.pkl      # Target semantic captions
│   ├── image_data.pkl           # Poster images (RGB uint8)
│   └── my_valid_data.txt        # Validation split (history → target)
│
├── pog_dataset/                 # Personalized outfit generation (POG)
│   ├── item_data.csv            # Item metadata
│   ├── outfit_data.csv          # Outfit → item mapping
│   ├── user_data.csv            # User histories
│   ├── captions.json            # Visual captions (Qwen2.5VL)
│   └── images.tar.gz            # Clothing item images
│
└── ser30k/                      # Sticker-based personalized generation
    ├── images/                  # All sticker images
    ├── captions.json            # Visual captions
    └── annotations/             # Original SER30K labels
```

---

# 🔍 Dataset Details

Below are usage guides for each major sub-dataset.

---

# 🎬 Movielens-MoviePoster Subset

## 1. Description

This subset supports **Personalized Movie Poster Generation**, pairing user rating histories with poster images and dual captions.

Includes:

- 100,836 user–movie interactions  
- 9,742 posters (RGB)  
- Movie metadata  
- Style and target captions  
- Predefined validation mapping  

---

## 2. File Definitions

### `info.csv`

| Field | Description |
|-------|-------------|
| id | Movie ID aligned with other files |
| name | Movie name |
| genre | Multi-label genre string (e.g., `"Action|Drama|Sci-Fi"`) |

---

### `ratings.csv`

| Field | Description |
|--------|-------------|
| userId | User ID |
| movieId | Movie ID |
| rating | Rating value |
| timestamp | Unix timestamp |

Sorting recommendation for history construction:

**rating ↓ → timestamp ↓**

---

### `style_captions.pkl`

- `dict[int, str or List[str]]`  
- Represents preference/style semantics  
- Lists are joined using commas  

---

### `target_captions.pkl`

- `dict[int, str or List[str]]`  
- Describes target movie semantics (characters, mood, visual cues)  

---

### `image_data.pkl`

- `dict[int, np.ndarray]`  
- Shape: `(H, W, 3)`  
- RGB, `uint8` images  

---

### `my_valid_data.txt`

Format:

```
<history_ids separated by ';'>,<target_id>
```

Example:

```
123;456;789,1011
```

---

# 👗 POG-Fashion Subset

## 1. Description

Used for **Personalized Outfit Generation** and multimodal recommendation explanation.

Contains:

- Item images  
- Title and descriptions  
- Outfit composition graph  
- User interaction histories  

---

## 2. File Definitions

### `item_data.csv`

| Field | Description |
|--------|-------------|
| item_id | Clothing item ID |
| cate_id | Category ID |
| title | Title text |
| desc | Full description |

---

### `outfit_data.csv`

| Field | Description |
|--------|-------------|
| outfit_id | Outfit ID |
| item_ids | Items forming the outfit |

---

### `user_data.csv`

| Field | Description |
|--------|-------------|
| user_id | User ID |
| item_history | Purchased / viewed items |
| outfit_history | Interacted outfits |

---

### `captions.json`

- Mapping: `item_id → Qwen2.5VL caption`  
- Captures color, texture, fabric, pattern, style cues  

---

# 😂 SER30K Sticker Subset

This subset uses SER30K sticker categories as pseudo-preference groups, enabling:

- emoji-style image generation  
- meme-style synthesis  
- personalized sticker recommendation  

Includes:

- sticker images  
- Qwen2.5VL captions  
- original annotations  

---

# 🛠 Usage Example

```python
import pickle, json
import pandas as pd

# Load movie style captions
style_caps = pickle.load(open("movieposter/style_captions.pkl", "rb"))

# Load outfit captions
with open("pog_dataset/captions.json") as f:
    outfit_caps = json.load(f)

# Load movie poster image
img_dict = pickle.load(open("movieposter/image_data.pkl", "rb"))
img = img_dict[1234]   # numpy array (H, W, 3)
```

---
