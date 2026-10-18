"""
Patches the Stage 2 cells in the Colab notebook to fix REDS path detection.
Run from the project root:
  python scripts/fix_notebook.py
"""

import json, os

NB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "VideoEnhancer_Training_Colab.ipynb")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── New cell sources ──────────────────────────────────────────────────────

CHECK_CELL_SRC = """\
# Check what REDS actually extracted
import os, glob

print('=== Checking REDS structure ===')
os.system('df -h /content')

all_pngs = glob.glob('data/REDS/**/*.png', recursive=True)
print(f'\\nTotal PNGs found under data/REDS: {len(all_pngs)}')
if all_pngs:
    print('Sample paths:')
    for p in all_pngs[:5]:
        print(f'  {p}')
else:
    print('WARNING: No PNGs found. REDS download may have failed or used a different path.')
    print('\\nContents of data/:')
    for root, dirs, files in os.walk('data'):
        depth = root.replace('data', '').count(os.sep)
        if depth > 3:
            continue
        print('  ' + '  ' * depth + os.path.basename(root) + '/')
"""

REDS_TRAIN_SRC = """\
import cv2
from torch.utils.data import Dataset, DataLoader
from enhancer.models.super_resolution import VideoSuperResolutionNet

class REDSDataset(Dataset):
    \"\"\"Builds T-frame sequences from REDS sharp clips with synthetic degradation.\"\"\"
    def __init__(self, reds_root, scale=4, patch_size=128, num_frames=5):
        self.scale = scale
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.sequences = []

        # Auto-detect clip dirs — REDS has several possible layouts:
        # A: data/REDS/train/train_sharp/000/*.png
        # B: data/REDS/train_sharp/000/*.png
        # C: flat folders directly containing PNGs
        candidate_patterns = [
            os.path.join(reds_root, 'train', 'train_sharp', '*'),
            os.path.join(reds_root, 'train_sharp', '*'),
            os.path.join(reds_root, 'sharp', '*'),
            os.path.join(reds_root, 'train', '*'),
            os.path.join(reds_root, '*'),
        ]

        clip_dirs = []
        for pattern in candidate_patterns:
            candidates = [d for d in glob.glob(pattern) if os.path.isdir(d)]
            if candidates:
                has_pngs = any(glob.glob(os.path.join(d, '*.png')) for d in candidates[:3])
                if has_pngs:
                    clip_dirs = candidates
                    print(f'REDS layout detected: {pattern}')
                    break

        if not clip_dirs:
            # Recursive fallback: find all dirs with enough PNGs
            all_pngs = glob.glob(os.path.join(reds_root, '**', '*.png'), recursive=True)
            parent_dirs = list(set(os.path.dirname(p) for p in all_pngs))
            clip_dirs = [d for d in parent_dirs
                         if len(glob.glob(os.path.join(d, '*.png'))) >= num_frames]
            if clip_dirs:
                print(f'Found {len(clip_dirs)} clip dirs via recursive search')

        for clip_dir in sorted(clip_dirs):
            frames = sorted(glob.glob(os.path.join(clip_dir, '*.png')))
            if len(frames) >= num_frames:
                for i in range(len(frames) - num_frames + 1):
                    self.sequences.append(frames[i:i+num_frames])

        print(f'REDSDataset: {len(clip_dirs)} clips -> {len(self.sequences)} sequences')
        if len(self.sequences) == 0:
            raise ValueError(
                f'No sequences found under {reds_root}. '
                f'Run the structure check cell to see what was extracted.'
            )

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        frame_paths = self.sequences[idx]
        frames_hr, frames_lr = [], []
        for fp in frame_paths:
            img = cv2.imread(fp)
            if img is None:
                img = np.zeros((128, 128, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            h, w = img.shape[:2]
            ps = self.patch_size
            if h < ps or w < ps:
                img = cv2.resize(img, (max(w, ps), max(h, ps)))
                h, w = img.shape[:2]
            y = np.random.randint(0, h - ps + 1)
            x = np.random.randint(0, w - ps + 1)
            crop_hr = img[y:y+ps, x:x+ps].copy()
            lr_ps   = ps // self.scale
            blur    = cv2.GaussianBlur(crop_hr, (3, 3), 0.5)
            crop_lr = cv2.resize(blur, (lr_ps, lr_ps), interpolation=cv2.INTER_CUBIC)
            crop_lr = np.clip(
                crop_lr + np.random.normal(
                    0, np.random.uniform(0, 10) / 255.0, crop_lr.shape
                ).astype(np.float32), 0, 1)
            frames_hr.append(torch.from_numpy(crop_hr).permute(2, 0, 1))
            frames_lr.append(torch.from_numpy(crop_lr).permute(2, 0, 1))
        return {
            'lr_seq': torch.stack(frames_lr, dim=0),
            'hr_ref': frames_hr[self.num_frames // 2],
        }

EPOCHS_VSR = 50
BATCH_VSR  = 8
LR_VSR     = 5e-5

video_model = VideoSuperResolutionNet(scale=4, num_frames=5, num_blocks=12).to(device)
vsr_dataset = REDSDataset('data/REDS', scale=4, patch_size=128, num_frames=5)
vsr_loader  = DataLoader(vsr_dataset, batch_size=BATCH_VSR, shuffle=True, num_workers=8, pin_memory=True)
vsr_opt     = optim.Adam(video_model.parameters(), lr=LR_VSR)
vsr_sched   = optim.lr_scheduler.CosineAnnealingLR(vsr_opt, T_max=EPOCHS_VSR)
vsr_loss_fn = nn.L1Loss()

print(f'Stage 2: Video SR | {len(vsr_dataset)} sequences | {EPOCHS_VSR} epochs | batch={BATCH_VSR}')
"""

# ── Find and replace the target cells ────────────────────────────────────

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

patched = 0
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))

    # Replace the structure-check cell (the one we added earlier)
    if "=== Checking REDS structure ===" in src:
        nb["cells"][i] = make_code_cell(CHECK_CELL_SRC)
        print(f"Patched cell {i}: REDS structure check")
        patched += 1

    # Replace the REDSDataset + training setup cell
    if "REDSDataset" in src and "VideoSuperResolutionNet" in src:
        nb["cells"][i] = make_code_cell(REDS_TRAIN_SRC)
        print(f"Patched cell {i}: REDSDataset + training setup")
        patched += 1

if patched == 0:
    print("WARNING: No target cells found. Check cell contents manually.")
else:
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"\nNotebook updated: {NB_PATH}")
    print(f"Patched {patched} cells.")
