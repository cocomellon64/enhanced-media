# Recommended Training Datasets

## Super-Resolution

| Dataset      | Frames  | Resolution | Use                          | URL |
|--------------|---------|------------|------------------------------|-----|
| DIV2K        | 800     | ~2K        | Image SR training            | https://data.vision.ee.ethz.ch/cvl/DIV2K/ |
| Flickr2K     | 2650    | ~2K        | Image SR augmentation        | https://cv.snu.ac.kr/research/EDSR/ |
| REDS         | 270     | 720p       | Video SR + deblur training   | https://seungjunnah.github.io/Datasets/reds.html |
| Vimeo-90K    | 91,701  | 448×256    | Video SR temporal training   | http://toflow.csie.ntu.edu.tw/ |

## Deblurring

| Dataset      | Description                     | URL |
|--------------|---------------------------------|-----|
| GoPro        | 3214 blur/sharp video pair clips | https://github.com/SeungjunNah/DeepDeblur-PyTorch |
| HIDE         | Human-aware deblur dataset       | https://github.com/joanshen0508/HA_deblur |

## Optical Flow

| Dataset         | Description                         | URL |
|-----------------|-------------------------------------|-----|
| Flying Chairs   | Synthetic chairs optical flow        | https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html |
| FlyingThings3D  | Synthetic 3D scene flow             | https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html |
| Sintel          | Optical flow benchmark               | http://sintel.is.tue.mpg.de/ |

## Held-Out Fidelity Validation Set

Use Set5, Set14, BSD100 for standard PSNR/SSIM benchmarking.

| Dataset | Frames | URL |
|---------|--------|-----|
| Set5    | 5      | Bundled with most SR frameworks |
| Set14   | 14     | Bundled with most SR frameworks |
| BSD100  | 100    | https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/ |

## Notes on Data Use

- Verify licenses before commercial use.
- Never include user-submitted media in training without explicit, informed consent and an opt-out mechanism.
- Maintain a hash of every training dataset version in the audit log.
