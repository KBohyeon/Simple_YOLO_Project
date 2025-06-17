
import sys
import os
import time
from pathlib import Path
import json

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from config import setup_directories, CLASSES, PATHS, PREDICT_CONFIG
    from predict_yolo import YOLOPredictor, check_model_file
    from PIL import Image
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"❌ 모듈 import 오류: {e}")
    print("필요한 파일들이 같은 폴더에 있는지 확인하세요:")
    print("  - config.py")
    print("  - predict_yolo.py")
    print("  - yolo_model.py")
    sys.exit(1)

def find_best_model():
    """최고 성능 모델 찾기"""
    models_dir = PATHS['models_dir']
    
    if not models_dir.exists():
        return None
    
    # 가능한 모델 파일들
    model_candidates = [
        models_dir / 'best_yolo_model.pth',
        models_dir / 'last_yolo_model.pth'
    ]
    
    # 다른 .pth 파일들도 찾기
    for model_file in models_dir.glob('*.pth'):
        if model_file not in model_candidates:
            model_candidates.append(model_file)
    
    # 존재하는 첫 번째 모델 반환
    for model_path in model_candidates:
        if model_path.exists():
            return model_path
    
    return None

def show_available_models():
    """사용 가능한 모델 목록 표시"""
    models_dir = PATHS['models_dir']
    
    if not models_dir.exists():
        print(f"❌ 모델 폴더가 없습니다: {models_dir}")
        return []
    
    model_files = list(models_dir.glob('*.pth'))
    
    if not model_files:
        print(f"❌ 훈련된 모델이 없습니다: {models_dir}")
        return []
    
    print(f"📁 사용 가능한 모델들:")
    for i, model_file in enumerate(model_files, 1):
        # 모델 정보 읽기 시도
        try:
            import torch
            checkpoint = torch.load(model_file, map_location='cpu')
            epoch = checkpoint.get('epoch', 'Unknown')
            loss = checkpoint.get('loss', 'Unknown')
            print(f"   {i}. {model_file.name} (에포크: {epoch}, 손실: {loss:.4f})")
        except:
            print(f"   {i}. {model_file.name}")
    
    return model_files

def select_test_images():
    """테스트 이미지 선택"""
    print(f"\n🖼️ 테스트 이미지 선택")
    print("=" * 40)
    
    # 기본 이미지 폴더에서 찾기
    images_dir = PATHS['images_dir']
    test_images = []
    
    if images_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            test_images.extend(images_dir.glob(ext))
            test_images.extend(images_dir.glob(ext.upper()))
    
    if test_images:
        print(f"   ✅ {len(test_images)}개의 이미지를 찾았습니다.")
        return test_images[:5]  # 처음 5개만 반환
    else:
        print(f"   ⚠️ 테스트 이미지가 없습니다.")
        return []

def interactive_prediction(predictor):
    """대화형 예측 모드"""
    print(f"\n🎮 대화형 예측 모드")
    print("=" * 40)
    print("명령어:")
    print("  - 이미지 경로 입력: 예측 실행")
    print("  - 'folder <폴더경로>': 폴더 내 모든 이미지 예측")
    print("  - 'test': 샘플 이미지로 테스트")
    print("  - 'config': 설정 변경")
    print("  - 'quit' 또는 'q': 종료")
    
    while True:
        try:
            command = input(f"\n명령 입력> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'q', 'exit']:
                break
            
            elif command.lower() == 'test':
                # 샘플 이미지 테스트
                test_images = select_test_images()
                if test_images:
                    image_path = test_images[0]
                    print(f"📸 테스트 이미지: {image_path.name}")
                    predictor.predict_image(image_path, save_result=True, show_result=True)
                else:
                    print("테스트 이미지가 없습니다.")
            
            elif command.lower() == 'config':
                # 설정 변경
                change_config(predictor)
            
            elif command.startswith('folder '):
                # 폴더 예측
                folder_path = command[7:].strip()
                if Path(folder_path).exists():
                    print(f"📁 폴더 예측: {folder_path}")
                    predictor.predict_batch(folder_path)
                else:
                    print(f"❌ 폴더가 없습니다: {folder_path}")
            
            else:
                # 이미지 파일 예측
                image_path = Path(command)
                if image_path.exists():
                    print(f"📸 이미지 예측: {image_path.name}")
                    predictor.predict_image(image_path, save_result=True, show_result=True)
                else:
                    print(f"❌ 파일이 없습니다: {image_path}")
        
        except KeyboardInterrupt:
            print(f"\n대화형 모드를 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {str(e)}")

def change_config(predictor):
    """예측 설정 변경"""
    print(f"\n⚙️ 현재 설정:")
    print(f"   - 신뢰도 임계값: {predictor.confidence_threshold}")
    print(f"   - NMS 임계값: {predictor.nms_threshold}")
    
    try:
        new_conf = input(f"새 신뢰도 임계값 (현재: {predictor.confidence_threshold}): ").strip()
        if new_conf:
            predictor.confidence_threshold = float(new_conf)
            print(f"✅ 신뢰도 임계값 변경: {predictor.confidence_threshold}")
    except ValueError:
        print("잘못된 값입니다.")
    
    try:
        new_nms = input(f"새 NMS 임계값 (현재: {predictor.nms_threshold}): ").strip()
        if new_nms:
            predictor.nms_threshold = float(new_nms)
            print(f"✅ NMS 임계값 변경: {predictor.nms_threshold}")
    except ValueError:
        print("잘못된 값입니다.")

def run_batch_demo(predictor):
    """배치 데모 실행"""
    print(f"\n📊 배치 예측 데모")
    
    test_images = select_test_images()
    
    if not test_images:
        print("데모용 이미지가 없습니다.")
        return
    
    print(f"   📸 {len(test_images)}개 이미지로 데모를 실행합니다.")
    
    results = []
    total_time = 0
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n[{i}/{len(test_images)}] {image_path.name}")
        
        try:
            detections, inference_time = predictor.predict_image(
                image_path, save_result=True, show_result=False
            )
            
            total_time += inference_time
            results.append({
                'image': image_path.name,
                'detections': len(detections),
                'time': inference_time
            })
            
        except Exception as e:
            print(f"   ❌ 실패: {e}")
    
    # 데모 결과 요약
    if results:
        avg_time = total_time / len(results)
        total_detections = sum(r['detections'] for r in results)
        
        print(f"\n📊 배치 데모 결과:")
        print(f"   - 처리된 이미지: {len(results)}개")
        print(f"   - 총 검출 객체: {total_detections}개")
        print(f"   - 평균 처리 시간: {avg_time*1000:.1f}ms")
        print(f"   - 이미지당 평균 객체: {total_detections/len(results):.1f}개")

def show_model_info(model_path):
    """모델 정보 표시"""
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"📋 모델 정보:")
        print(f"   - 파일: {model_path.name}")
        print(f"   - 훈련 에포크: {checkpoint.get('epoch', 'Unknown')}")
        print(f"   - 최종 손실: {checkpoint.get('loss', 'Unknown'):.4f}")
        print(f"   - 클래스: {', '.join(checkpoint.get('classes', CLASSES))}")
        
        # 파일 크기
        file_size = model_path.stat().st_size / (1024*1024)
        print(f"   - 파일 크기: {file_size:.1f}MB")
        
        # 수정 시간
        import datetime
        mtime = datetime.datetime.fromtimestamp(model_path.stat().st_mtime)
        print(f"   - 수정 시간: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"모델 정보를 읽을 수 없습니다: {e}")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🔍 YOLO 객체 검출 예측 시스템")
    print("=" * 60)
    
    # 디렉토리 설정
    setup_directories()
    
    print(f"🎯 검출 클래스: {', '.join(CLASSES)}")
    
    # 모델 찾기
    print(f"\n🔍 훈련된 모델 검색 중...")
    model_path = find_best_model()
    
    if not model_path:
        print(f"❌ 훈련된 모델이 없습니다!")
        show_available_models()
        print(f"\n💡 해결 방법:")
        print(f"1. python run_training.py 실행하여 모델 훈련")
        print(f"2. 기존 모델 파일을 {PATHS['models_dir']}에 복사")
        return
    
    print(f"✅ 모델 발견: {model_path.name}")
    show_model_info(model_path)
    
    # 예측기 초기화
    try:
        print(f"\n🔄 예측 시스템 초기화 중...")
        predictor = YOLOPredictor(
            model_path=model_path,
            confidence_threshold=PREDICT_CONFIG['confidence_threshold'],
            nms_threshold=PREDICT_CONFIG['nms_threshold']
        )
        
        print(f"✅ 예측 시스템 준비 완료!")
        
    except Exception as e:
        print(f"❌ 예측 시스템 초기화 실패: {str(e)}")
        return
    
    # 메뉴 표시
    print(f"\n📋 사용 가능한 기능:")
    print(f"1. 대화형 예측 모드")
    print(f"2. 배치 데모 실행")
    print(f"3. 성능 분석")
    print(f"4. 종료")
    
    while True:
        try:
            choice = input(f"\n선택 (1-4): ").strip()
            
            if choice == '1':
                interactive_prediction(predictor)
            
            elif choice == '2':
                run_batch_demo(predictor)
            
            elif choice == '3':
                test_images = select_test_images()
                if test_images:
                    # 임시 폴더에 복사해서 성능 분석
                    temp_dir = PATHS['base_dir'] / 'temp_test'
                    temp_dir.mkdir(exist_ok=True)
                    
                    for img in test_images[:3]:  # 처음 3개만
                        import shutil
                        shutil.copy(img, temp_dir)
                    
                    predictor.evaluate_performance(temp_dir)
                    
                    # 임시 폴더 정리
                    shutil.rmtree(temp_dir)
                else:
                    print("성능 분석할 이미지가 없습니다.")
            
            elif choice == '4':
                print("예측 시스템을 종료합니다.")
                break
            
            else:
                print("잘못된 선택입니다. 1-4 중에서 선택하세요.")
        
        except KeyboardInterrupt:
            print(f"\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {str(e)}")

if __name__ == "__main__":
    main()