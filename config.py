# config.py - 공통 설정 파일

import torch
from pathlib import Path

# 클래스 정의 (고양이와 사람만)
CLASSES = ['cat', 'person']
NUM_CLASSES = len(CLASSES)

# 모델 하이퍼파라미터
MODEL_CONFIG = {
    'num_classes': NUM_CLASSES,
    'grid_size': 7,
    'img_size': 224,
    'output_size': 5 + NUM_CLASSES  # x, y, w, h, conf + classes
}

# 훈련 하이퍼파라미터
TRAIN_CONFIG = {
    'epochs': 30,
    'batch_size': 4,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'lambda_coord': 5.0,
    'lambda_noobj': 0.5,
    'train_val_split': 0.8
}

# 예측 설정
PREDICT_CONFIG = {
    'confidence_threshold': 0.3,
    'nms_threshold': 0.5
}

# 파일 경로
PATHS = {
    'base_dir': Path('yolo_dataset'),
    'images_dir': Path('yolo_dataset/images'),
    'annotations_dir': Path('yolo_dataset/annotations'),
    'models_dir': Path('yolo_dataset/models'),
    'results_dir': Path('yolo_dataset/results')
}

# 색상 (시각화용)
COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# 디바이스 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 변환 설정
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

def setup_directories():
    """필요한 디렉토리 생성"""
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
    
    print("📁 디렉토리 구조:")
    for name, path in PATHS.items():
        print(f"   - {name}: {path}")

if __name__ == "__main__":
    setup_directories()
    print(f"\n🎯 클래스: {CLASSES}")
    print(f"🖥️ 디바이스: {DEVICE}")