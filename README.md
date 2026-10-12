# Group 98 – Flower classification with frozen features + KNN

## Structure
```
flower-knn/
├── extract_features.py   # step 1: images -> feature vectors (GPU helps)
├── knn_eval.py           # step 2: KNN (+ linear) on saved features (CPU)
├── requirements.txt
├── data/                 # datasets (not in git)
├── features/             # saved .npz features (not in git)
└── results/              # json results + confusion matrices
```

## Setup (once per computer)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

DINOv3 is gated on Hugging Face:
1. Create a Hugging Face account and accept the licence on
   https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
2. Create an access token (Settings -> Access Tokens) and run
   `huggingface-cli login` and paste the token.

## Data
Download the Kaggle dataset (marquis03/flower-classification), unzip it and
put it so that it looks like:
```
data/flowers/train/<class>/*.jpg
data/flowers/val/<class>/*.jpg
```
Rename the folders if the zip uses other names.

## Run
```bash
# Quick test on 20 images per split (works on CPU)
python extract_features.py --data-root data/flowers --model resnet50 --limit 20

# Full extraction (preferably on the GPU VM)
python extract_features.py --data-root data/flowers --model resnet50
python extract_features.py --data-root data/flowers --model dinov3

# Optional reference point: random weights
python extract_features.py --data-root data/flowers --model resnet50 --random-init

# Evaluation (CPU, seconds)
python knn_eval.py --dataset flowers --model resnet50 --linear
python knn_eval.py --dataset flowers --model dinov3 --linear
```
