# MoodLens

![Banner](assets/pic.png)

MoodLens is a small computer vision and deep learning project for facial emotion recognition. It detects faces in video streams or images using Haar Cascades and describes their emotion using a compact CNN.

This project is designed for educational use and modular experimentation. The app comes with a preloaded and trained model, but you can also train your own using the FER2013 dataset. 

The codebase is organized into focused modules for data handling, model definition, training routines, face detection, and a real-time user interface.

---

## Table of contents

- [Project structure](#project-structure)
- [Setup](#setup)
- [Dataset](#dataset)
- [Prepare splits](#prepare-splits)
- [Training](#training)
- [Run real‑time webcam demo](#Webcam-demo)
- [Configuration reference](#configuration-reference)
- [Model architecture](#model-architecture)
- [Performance & timing](#performance--timing)



## Project structure

```
emotion_vision/
├── datasets/                  # generated after prepare_fer2013
│   ├── train.csv              # generated
│   ├── val.csv                # generated
│   ├── test_public.csv        # generated
│   └── test_private.csv       # generated
│
├── models/                    # saved model checkpoints
│   └── best.pt                # best checkpoint (state dict)
│
├── assets/                    
│   └── haarcascade_frontalface_default.xml
│
├── src/
│   ├── data/
│   │   ├── fer2013.csv            # original FER2013 dataset CSV - download from Kaggle
│   │   ├── fer2013.py             # FER2013 dataset class + transforms
│   │   └── prepare_fer2013.py     # prepare train/val/test CSVs (no argparse)
│   │
│   ├── models/
│   │   ├── CNN_class.py           # compact CNN (activation configurable, ELU default + SE)
│   │   └── factory.py             # create_model(name, **kwargs)
│   │
│   ├── realtime/
│   │   ├── app.py                 # webcam loop (display boxes + labels)
│   │   ├── HaarDetector.py          # HaarFaceDetector wrapper
│   │   └── run_webcam.py          # run_webcam(model, cascade, ...)
│   │
│   ├── train/
│   │   ├── train_loop.py          # TrainConfig + train_model(...)
│   │   ├── train_config.py        # default config values
│   │   ├── config.py              # global constants (classes, img size)
│   │   ├── train.py               # entrypoint script (no argparse, tweak defaults inside)
│   │   └── estimate_train.py      # measure epoch time/estimate total
│   │
│   ├── test/
│   │   └── test.py                # evaluate model on test split
│   │
│   └── utils/
│       ├── io.py                  # helper for saving/loading
│       ├── seed.py                # fix_seed()
│       └── time_measure.py        # timing utilities
│
├── pyproject.toml                 # optional (project metadata)
├── requirements.txt               # pinned dependencies
└── README.md                      # usage instructions      
```

> If your local tree differs, the import paths still assume the `src/` package layout shown above.

---

## Setup

- **Python**: 3.9+
- **OS**: Windows / macOS / Linux
- **GPU**: Optional (project runs fine on CPU).

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

With the last one, you will install these packages:

- torch
- torchvision
- opencv-python
- numpy
- pandas
- matplotlib
- tqdm

---

## Dataset

This project uses the **FER2013** CSV. Obtain `fer2013.csv` from
the [Kaggle FER2013 dataset](https://www.kaggle.com/datasets/deadskull7/fer2013).

After downloading, place it in:

```
src/data/fer2013.csv
```

FYI, the dataset uses the standard 7 classes in this order:

```
Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral
```

---

## Prepare splits

Generate **train**, **val**, and **test** CSVs (no images extracted; still CSV‑based):

```bash
# Run as a module (recommended to avoid import issues):
python -m src.data.prepare_fer2013

# or, if you prefer direct:
python src/data/prepare_fer2013.py
```

But for the latter, ensure you run it from the project root directory (where `src/` is located) to avoid import errors
and don`t include __init__.py files directories.

The script reads `src/data/fer2013.csv` and writes:

```
datasets/train.csv
datasets/val.csv
datasets/test_public.csv
datasets/test_private.csv
```

A 10% validation split is carved from the original training partition with a fixed seed for reproducibility.

---

## Training

Open `src/train/train.py` to tweak the defaults (epochs, batch size, workers, etc.), then run:

```bash
# Safer (packages imports correctly)
python -m src.train.train

# or direct
python src/train/train.py
```

**What gets saved?**  
The best checkpoint is written to:

```
models/best.pt
```

It contains `state["model"]` and `state["classes"]`. You can change the output directory via `OUT_DIR` in `train.py`.

---

## Testing

Evaluate the trained model on the test split:

```bash
python -m src.test.eval_test

# or

python src/test/test.py
```

It loads the best checkpoint from `models/best.pt` by default. Adjust the path in `test.py` if needed.
It uses the Public test split (`datasets/test_public.csv`).

---

## Webcam demo

1. Make sure you have a trained model at `models/fer2013/cnn_small/best.pt` (or adjust the path below).
2. Download `haarcascade_frontalface_default.xml` and place it in `assets/` (or point to OpenCV’s built‑in path).

Run:

```bash
python -m src.realtime.run_webcam
# or
python src/realtime/run_webcam.py
```

**Tips**

- Press `Esc` or `q` to quit.
- On some machines, CPU inference gives lower latency than GPU for webcam use. You can switch device inside
  `realtime/run_webcam.py`

---

## Configuration reference

### Training (`TrainConfig` in `train_loop.py`)

- `epochs` (default 25)
- `lr` (default 1e-3)
- `weight_decay` (default 1e-5)
- `batch_size` (e.g., 128–512; adjust to RAM)
- `num_workers` (2–8 is a good CPU range)
- `prefetch_factor`, `persistent_workers`
- `device` (`"cuda"` if available, else `"cpu"`)
- `out_dir` (where checkpoints are saved)
- Early stopping: `early_stop_patience`, `early_stop_delta`
- `channels_last` (for potential CPU throughput gains)
- `use_torch_compile` (PyTorch 2.x)
- `show_progress` (tqdm progress bars)

### Inference (webcam)

- `MODEL_PATH` — defaults to `model/best.pt`
- `CASCADE_PATH` — path to Haar cascade XML
- `DEVICE` — `"cpu"` or `"cuda"`
- `USE_CLAHE` — apply CLAHE preprocessing
- `DETECT_EVERY_N` — run face detector every N frames

### Defaults and labels

- Default classes: `["Anger","Disgust","Fear","Happy","Sad","Surprise","Neutral"]`

---

## Model architecture

A compact CNN for 48×48 grayscale inputs:

- Convolutional blocks with **BatchNorm** and **ELU** activations.
- Light **SE‑style** channel attention before the classifier head.
- **Dropout** in the trunk and head to regularize.
- Final linear head → logits over 7 classes.

This design aims to be small, fast, and robust on FER2013.

---

## Performance & timing

Curious how long training might take on your machine? Use the micro‑benchmark:

```bash
python -m src.train.eta_probe
# or
python src/train/estimate_train.py
```

It warms up, times a few train/val steps, and prints an estimated **time per epoch** and **total** for your configured
number of epochs.