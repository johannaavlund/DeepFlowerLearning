from pathlib import Path
import shutil
from scipy.io import loadmat

ROOT = Path("data/flowers102_raw")
IMAGE_DIR = ROOT / "jpg"
OUT_DIR = Path("data/flowers102")

labels = loadmat(ROOT / "imagelabels.mat")["labels"].ravel()
split_data = loadmat(ROOT / "setid.mat")

splits = {
    "train": split_data["trnid"].ravel(),
    "val": split_data["valid"].ravel(),
    "test": split_data["tstid"].ravel(),
}

for split_name, image_ids in splits.items():
    count = 0
    for image_id in image_ids:
        image_id = int(image_id)
        label = int(labels[image_id - 1])
        src = IMAGE_DIR / f"image_{image_id:05d}.jpg"
        dst_dir = OUT_DIR / split_name / f"class_{label:03d}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        if not src.exists():
            raise FileNotFoundError(src)
        if not dst.exists():
            shutil.copy2(src, dst)
        count += 1

    print(f"{split_name}: {count} images")

print(f"\nPrepared dataset at {OUT_DIR}")
