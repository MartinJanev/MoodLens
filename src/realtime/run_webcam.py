# Run webcam inference with fixed defaults (no argparse).
from src.realtime.app import run_webcam

# ======= Defaults you can tweak =======
MODEL_PATH = 'models/fer2013/cnn_small/best.pt'  # path to the trained model
CASCADE_PATH = 'assets/haarcascade_frontalface_default.xml'
DEVICE = 'cpu'  # 'cpu' or 'cuda'
USE_CLAHE = True  # preprocessing
CAMERA_INDEX = 0  # 0 = default camera
DETECT_EVERY_N = 2  # run detector every N frames for speed
# ======================================

if __name__ == '__main__':
    run_webcam(
        model_path=MODEL_PATH,
        cascade_path=CASCADE_PATH,
        device=DEVICE,
        use_clahe=USE_CLAHE,
        camera_index=CAMERA_INDEX,
        detect_every_n=DETECT_EVERY_N,
    )
