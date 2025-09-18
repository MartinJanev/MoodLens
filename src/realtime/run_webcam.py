# Run webcam inference with fixed defaults (no argparse).
import torch

from src.realtime.app import run_webcam

# ======= Defaults you can tweak =======
MODEL_PATH = 'models/cnn_small/model.pt'  # path to the trained model
MODEL_NAME = 'resnet18'  # 'cnn_small' or 'resnet18'
CASCADE_PATH = 'assets/haarcascade_frontalface_default.xml'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_CLAHE = True  # preprocessing
CAMERA_INDEX = 0  # 0 = default camera
DETECT_EVERY_N = 2  # run detector every N frames for speed

RESNET_INPUT_SIZE = 160  # 160 is a good speed/accuracy trade-off for ResNet

# ======================================

def main():
    run_webcam(
        model_path=MODEL_PATH,
        cascade_path=CASCADE_PATH,
        device=DEVICE,
        use_clahe=USE_CLAHE,
        camera_index=CAMERA_INDEX,
        detect_every_n=DETECT_EVERY_N,
        model_name=MODEL_NAME,
        resnet_input_size=RESNET_INPUT_SIZE,
    )

if __name__ == '__main__':
    main()
