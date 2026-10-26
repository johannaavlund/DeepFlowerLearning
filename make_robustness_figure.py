from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import random
import matplotlib.pyplot as plt

# Use a real image from the official test split.
root = Path("data/flowers102/test")
candidates = sorted(root.glob("class_*/*.jpg"))
if not candidates:
    raise FileNotFoundError(f"No test JPG images found in {root}")

# Pick a reproducible image with a reasonably large original resolution.
image_path = candidates[0]
im = Image.open(image_path).convert("RGB")

# Match the experiment: Resize(256), perturb, then CenterCrop(224).
im = ImageOps.contain(im, (256, 256), method=Image.Resampling.BICUBIC)

def center_crop_224(image):
    w, h = image.size
    left, top = (w - 224) // 2, (h - 224) // 2
    return image.crop((left, top, left + 224, top + 224))

# The five perturbations match extract_features.py.
gray = im.convert("L").convert("RGB")
rotated = im.rotate(90, expand=True)
dark = ImageEnhance.Brightness(im).enhance(0.4)
blurred = im.filter(ImageFilter.GaussianBlur(radius=4))

w, h = im.size
box = (w // 4, h // 4, w - w // 4, h - h // 4)
cropped = im.crop(box).resize((w, h), Image.Resampling.BICUBIC)

# Apply the same final center crop as the experiment.
panels = [
    ("Original", center_crop_224(im)),
    ("Grayscale", center_crop_224(gray)),
    ("Rotated 90°", center_crop_224(rotated)),
    ("Darkened", center_crop_224(dark)),
    ("Gaussian blur", center_crop_224(blurred)),
    ("Zoom crop", center_crop_224(cropped)),
]

# Poster-ready layout.
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.patch.set_facecolor("white")

for ax, (label, image) in zip(axes.flat, panels):
    ax.imshow(image)
    ax.set_title(label, fontsize=16, fontweight="bold", pad=10)
    ax.axis("off")

fig.suptitle(
    "Robustness to Image Distortions",
    fontsize=22, fontweight="bold", y=0.98
)
fig.text(
    0.5, 0.025,
    "One Flowers-102 test image shown under five image perturbations",
    ha="center", fontsize=12
)
plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.93])

out = Path("results/flowers102_full/robustness_example.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Source image: {image_path}")
print(f"Saved figure: {out}")
