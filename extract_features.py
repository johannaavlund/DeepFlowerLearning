"""
extract_features.py
-------------------
Step 1 of the project: run every image through a FROZEN pretrained backbone
(ResNet-50 or DINOv3) once, and save the resulting feature vectors to disk.
All later experiments (KNN, linear classifier, few-shot, t-SNE) only use
these saved features, so this is the only step that benefits from a GPU.

Expected data layout (standard "ImageFolder" layout, one folder per class):

    data/flowers/
        train/  <class_a>/img1.jpg  <class_b>/...
        val/    <class_a>/...       <class_b>/...
        test/   (optional)

Usage examples:
    python extract_features.py --data-root data/flowers --model resnet50
    python extract_features.py --data-root data/flowers --model dinov3
    python extract_features.py --data-root data/flowers --model dinov3 --limit 20   # quick test

    # Grayscale experiment: extracts only the test images, converted to grayscale
    python extract_features.py --data-root data/flowers --model resnet50 --grayscale
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# Hugging Face model id for DINOv3 ViT-B/16 (gated: accept the licence on the
# model page and run `huggingface-cli login` once before using it).
DINOV3_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# Both backbones expect ImageNet mean/std normalisation.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_backbone(name: str, pretrained: bool, device: torch.device):
    """Return (model, transform) for the requested backbone.

    The model maps a batch of images to one feature vector per image.
    """
    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        # Remove the final classification layer -> 2048-d pooled features.
        model.fc = torch.nn.Identity()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        forward = lambda x: model(x)

    elif name == "dinov3":
        from transformers import AutoConfig, AutoModel

        if pretrained:
            model = AutoModel.from_pretrained(DINOV3_ID)
        else:
            # Same architecture, random weights (reference point only).
            model = AutoModel.from_config(AutoConfig.from_pretrained(DINOV3_ID))
        # 224x224 = 14x14 patches of size 16.
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        # pooler_output = the CLS token embedding (768-d for ViT-B).
        forward = lambda x: model(pixel_values=x).pooler_output

    else:
        raise ValueError(f"Unknown model: {name}")

    model.eval().to(device)
    return forward, transform


def find_splits(data_root: str):
    """Return the split folders (train/val/test) that exist under data_root."""
    splits = [s for s in ("train", "val", "test")
              if os.path.isdir(os.path.join(data_root, s))]
    if "train" not in splits:
        raise FileNotFoundError(
            f"No 'train' folder found in {data_root}. "
            "Check the data layout described at the top of this file.")
    return splits


@torch.no_grad()
def extract(forward, loader, device):
    """Run all batches through the backbone and collect features + labels."""
    feats, labels = [], []
    for images, targets in tqdm(loader, desc="  batches"):
        images = images.to(device, non_blocking=True)
        out = forward(images)
        feats.append(out.float().cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", choices=["resnet50", "dinov3"], required=True)
    parser.add_argument("--random-init", action="store_true",
                        help="Use randomly initialised weights instead of pretrained.")
    parser.add_argument("--out-dir", default="features")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only use N random images per split (for quick tests).")
    parser.add_argument("--grayscale", action="store_true",
                        help="Extract grayscale versions of the TEST images only "
                             "(for the colour-dependence experiment).")
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)

    # Pick GPU if available, then Apple Silicon, otherwise CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    forward, transform = build_backbone(args.model, not args.random_init, device)

    if args.grayscale:
        # Convert every image to grayscale (copied to 3 channels so the
        # networks still get the input shape they expect). Everything else
        # in the preprocessing stays the same.
        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3)] + transform.transforms)

    dataset_name = os.path.basename(os.path.normpath(args.data_root))
    tag = f"{args.model}{'_random' if args.random_init else ''}"
    os.makedirs(args.out_dir, exist_ok=True)

    # Which splits to extract. In grayscale mode only the TEST images are
    # needed (the KNN reference set stays in colour):
    #   - dataset with a 'test' folder (Flowers-102): extract 'test' only
    #   - otherwise (Kaggle-14): extract only the held-out test images listed
    #     in features/<dataset>_split.npz (created by knn_eval.py)
    splits = find_splits(args.data_root)
    keep_paths = None
    if args.grayscale:
        if "test" in splits:
            splits = ["test"]
        else:
            split_file = os.path.join(args.out_dir, f"{dataset_name}_split.npz")
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"{split_file} not found. Run knn_eval.py first.")
            keep_paths = set(np.load(split_file)["test_paths"])
            splits = ["train"]

    class_names = None
    for split in splits:
        ds = datasets.ImageFolder(os.path.join(args.data_root, split), transform=transform)

        # Make sure every split uses the same class -> index mapping as train.
        if class_names is None:
            class_names = ds.classes
        elif ds.classes != class_names:
            raise ValueError(f"Classes in '{split}' differ from 'train'.")

        paths = [p for p, _ in ds.samples]
        if keep_paths is not None:
            idx = [i for i, p in enumerate(paths) if p in keep_paths]
            if not idx:
                raise ValueError("No test images matched the saved split. Use the "
                                 "same --data-root as when the split was created.")
            ds = Subset(ds, idx)
            paths = [paths[i] for i in idx]
        if args.limit:
            idx = random.sample(range(len(ds)), min(args.limit, len(ds)))
            ds = Subset(ds, idx)
            paths = [paths[i] for i in idx]

        print(f"[{split}] {len(ds)} images, {len(class_names)} classes")
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")
        feats, labels = extract(forward, loader, device)

        if args.grayscale:
            out_path = os.path.join(args.out_dir, f"{dataset_name}_{tag}_gray_test.npz")
        else:
            out_path = os.path.join(args.out_dir, f"{dataset_name}_{tag}_{split}.npz")
        np.savez_compressed(out_path, features=feats, labels=labels,
                            paths=np.array(paths), class_names=np.array(class_names))
        print(f"  saved {feats.shape} -> {out_path}")


if __name__ == "__main__":
    main()