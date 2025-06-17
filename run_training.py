
import sys
import os
import json
import time
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from config import setup_directories, CLASSES, PATHS, TRAIN_CONFIG, DEVICE
    from train_yolo import YOLOTrainer, check_dataset
    import torch
except ImportError as e:
    print(f"❌ 모듈 import 오류: {e}")
    print("필요한 파일들이 같은 폴더에 있는지 확인하세요:")
    print("  - config.py")
    print("  - train_yolo.py")
    print("  - yolo_model.py")
    sys.exit(1)

def check_requirements():
    """필수 라이브러리 및 환경 확인"""
    print("🔍 시스템 환경 확인 중...")
    
    # PyTorch 확인
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        print(f"   ✅ CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ CUDA 버전: {torch.version.cuda}")
    except ImportError:
        print("   ❌ PyTorch가 설치되지 않았습니다.")
        print("   설치: pip install torch torchvision")
        return False
    
    # 기타 라이브러리 확인
    required_packages = {
        'matplotlib': 'matplotlib',
        'PIL': 'pillow',
        'numpy': 'numpy'
    }
    
    for package_name, install_name in required_packages.items():
        try:
            if package_name == 'PIL':
                from PIL import Image
            elif package_name == 'matplotlib':
                import matplotlib.pyplot as plt
            elif package_name == 'numpy':
                import numpy as np
            print(f"   ✅ {package_name}: 설치됨")
        except ImportError:
            print(f"   ❌ {package_name}: 미설치")
            print(f"   설치: pip install {install_name}")
            return False
    
    return True

def interactive_config():
    """대화형 훈련 설정"""
    print("\n⚙️ 훈련 설정")
    print("=" * 40)
    
    config = TRAIN_CONFIG.copy()
    
    # 에포크 수
    try:
        epochs = input(f"에포크 수 (현재: {config['epochs']}): ").strip()
        if epochs:
            config['epochs'] = int(epochs)
    except ValueError:
        print("잘못된 입력입니다. 기본값을 사용합니다.")
    
    # 배치 크기
    try:
        batch_size = input(f"배치 크기 (현재: {config['batch_size']}): ").strip()
        if batch_size:
            config['batch_size'] = int(batch_size)
    except ValueError:
        print("잘못된 입력입니다. 기본값을 사용합니다.")
    
    # 학습률
    try:
        lr = input(f"학습률 (현재: {config['learning_rate']}): ").strip()
        if lr:
            config['learning_rate'] = float(lr)
    except ValueError:
        print("잘못된 입력입니다. 기본값을 사용합니다.")
    
    return config

def estimate_training_time(num_images, epochs, batch_size):
    """훈련 시간 추정"""
    # GPU/CPU에 따른 대략적인 시간 추정
    if DEVICE.type == 'cuda':
        time_per_image = 0.1  # GPU: 이미지당 0.1초
    else:
        time_per_image = 0.5  # CPU: 이미지당 0.5초
    
    total_time = (num_images * epochs * time_per_image) / batch_size
    
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    if hours > 0:
        return f"약 {hours}시간 {minutes}분"
    else:
        return f"약 {minutes}분"

def show_training_tips():
    """훈련 팁 표시"""
    print("\n💡 훈련 팁:")
    print("   • 각 클래스당 최소 50개 이상의 객체를 라벨링하세요")
    print("   • GPU 사용시 배치 크기를 늘려 훈련 속도를 높일 수 있습니다")
    print("   • 메모리 부족시 배치 크기를 줄이세요 (2 또는 1)")
    print("   • 훈련 중 Ctrl+C로 중단 가능합니다")
    print("   • 훈련 과정은 실시간 그래프로 모니터링됩니다")

def check_disk_space():
    """디스크 공간 확인"""
    try:
        import shutil
        free_space = shutil.disk_usage(PATHS['base_dir']).free
        free_gb = free_space / (1024**3)
        
        if free_gb < 1:
            print(f"⚠️ 디스크 공간 부족: {free_gb:.1f}GB")
            print("최소 1GB 이상의 여유 공간이 필요합니다.")
            return False
        else:
            print(f"   ✅ 디스크 여유 공간: {free_gb:.1f}GB")
            return True
    except:
        return True

def save_training_config(config, model_path):
    """훈련 설정 저장"""
    config_file = PATHS['models_dir'] / 'training_config.json'
    
    config_data = {
        'training_config': config,
        'model_path': str(model_path),
        'classes': CLASSES,
        'device': str(DEVICE),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print(f"📝 훈련 설정 저장: {config_file}")

def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("🚀 YOLO 모델 훈련 시스템")
    print("=" * 70)
    
    # 시스템 요구사항 확인
    if not check_requirements():
        return
    
    # 디스크 공간 확인
    if not check_disk_space():
        return
    
    # 디렉토리 설정
    setup_directories()
    
    print(f"\n🎯 훈련 클래스: {', '.join(CLASSES)}")
    print(f"🖥️ 디바이스: {DEVICE}")
    print(f"📂 데이터 폴더: {PATHS['images_dir']}")
    
    # 데이터셋 확인
    print(f"\n📊 데이터셋 확인 중...")
    if not check_dataset(PATHS['images_dir']):
        print(f"\n❌ 훈련을 시작할 수 없습니다.")
        print(f"먼저 다음을 수행하세요:")
        print(f"1. python run_labeling.py 실행")
        print(f"2. 이미지 라벨링 완료")
        print(f"3. 각 클래스당 최소 20개 이상 라벨링")
        return
    
    # 훈련 설정
    print(f"\n⚙️ 기본 훈련 설정:")
    for key, value in TRAIN_CONFIG.items():
        print(f"   - {key}: {value}")
    
    # 대화형 설정 변경
    use_custom = input(f"\n설정을 변경하시겠습니까? (y/N): ")
    if use_custom.lower() in ['y', 'yes']:
        config = interactive_config()
    else:
        config = TRAIN_CONFIG.copy()
    
    # 훈련 시간 추정
    from train_yolo import RealImageDataset
    try:
        temp_dataset = RealImageDataset(PATHS['images_dir'])
        num_images = len(temp_dataset)
        estimated_time = estimate_training_time(num_images, config['epochs'], config['batch_size'])
        print(f"\n⏱️ 예상 훈련 시간: {estimated_time}")
    except:
        print(f"\n⏱️ 훈련 시간을 추정할 수 없습니다.")
    
    # 훈련 팁 표시
    show_training_tips()
    
    # 최종 확인
    print(f"\n📋 최종 훈련 설정:")
    for key, value in config.items():
        print(f"   - {key}: {value}")
    
    response = input(f"\n훈련을 시작하시겠습니까? (Y/n): ")
    if response.lower() not in ['', 'y', 'yes']:
        print("훈련이 취소되었습니다.")
        return
    
    try:
        print(f"\n🚀 훈련 시작!")
        print("=" * 50)
        
        start_time = time.time()
        
        # 훈련 설정 저장
        model_path = PATHS['models_dir'] / 'best_yolo_model.pth'
        save_training_config(config, model_path)
        
        # 훈련 실행
        trainer = YOLOTrainer(PATHS['images_dir'], config)
        trainer.train()
        
        # 훈련 완료 처리
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 훈련이 성공적으로 완료되었습니다!")
        print(f"⏱️ 실제 훈련 시간: {total_time//3600:.0f}시간 {(total_time%3600)//60:.0f}분")
        print(f"💾 모델 저장 위치: {PATHS['models_dir']}")
        
        # 다음 단계 안내
        print(f"\n📋 다음 단계:")
        print(f"1. 모델 테스트: python run_prediction.py")
        print(f"2. 단일 이미지 예측: python predict_yolo.py --image test.jpg")
        print(f"3. 폴더 배치 예측: python predict_yolo.py --folder test_images/")
        
        # 자동으로 예측 실행 여부 확인
        test_now = input(f"\n지금 바로 예측을 테스트해보시겠습니까? (y/N): ")
        if test_now.lower() in ['y', 'yes']:
            print(f"\n🔍 예측 시스템 실행 중...")
            try:
                # 예측 스크립트 실행
                import subprocess
                subprocess.run([sys.executable, 'run_prediction.py'])
            except Exception as e:
                print(f"예측 실행 실패: {e}")
                print(f"수동으로 실행하세요: python run_prediction.py")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 사용자에 의해 훈련이 중단되었습니다.")
        print(f"부분적으로 훈련된 모델이 저장되었을 수 있습니다.")
        
    except Exception as e:
        print(f"\n❌ 훈련 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🛠️ 문제 해결 방법:")
        print(f"1. 메모리 부족: 배치 크기를 줄이세요 (--batch-size 2)")
        print(f"2. CUDA 오류: CPU 모드로 실행하세요")
        print(f"3. 데이터 문제: 라벨링을 다시 확인하세요")

if __name__ == "__main__":
    main()