

import sys
import os
import subprocess
from pathlib import Path
import time

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from config import setup_directories, CLASSES, PATHS
    import json
except ImportError as e:
    print(f"❌ 모듈 import 오류: {e}")
    print("config.py 파일이 있는지 확인하세요.")
    sys.exit(1)

def show_banner():
    """시스템 배너 표시"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║                🎯 YOLO 객체 검출 시스템                       ║
║                                                              ║
║     실제 이미지로 훈련 가능한 간단한 YOLO 구현체              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"🎯 검출 클래스: {', '.join(CLASSES)}")
    print(f"📂 작업 폴더: {PATHS['base_dir']}")

def check_system_status():
    """시스템 상태 확인"""
    print(f"\n🔍 시스템 상태 확인")
    print("=" * 50)
    
    status = {
        'directories': False,
        'images': 0,
        'labeled_images': 0,
        'models': 0,
        'results': 0
    }
    
    # 디렉토리 확인
    try:
        setup_directories()
        status['directories'] = True
        print(f"   ✅ 디렉토리 구조: 정상")
    except Exception as e:
        print(f"   ❌ 디렉토리 구조: 오류 ({e})")
    
    # 이미지 확인
    if PATHS['images_dir'].exists():
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(PATHS['images_dir'].glob(ext))
            image_files.extend(PATHS['images_dir'].glob(ext.upper()))
        
        status['images'] = len(image_files)
        print(f"   📸 이미지 파일: {status['images']}개")
        
        # ★ 핵심 수정 부분 ★
        labeled_count = 0
        if PATHS['annotations_dir'].exists():
            for img_file in image_files:
                json_file = PATHS['annotations_dir'] / f"{img_file.stem}.json"
                if json_file.exists():
                    labeled_count += 1
        
        status['labeled_images'] = labeled_count
        print(f"   🏷️ 라벨링 완료: {status['labeled_images']}개")
    
    # 모델 확인
    if PATHS['models_dir'].exists():
        model_files = list(PATHS['models_dir'].glob('*.pth'))
        status['models'] = len(model_files)
        print(f"   🧠 훈련된 모델: {status['models']}개")
    
    # 결과 확인
    if PATHS['results_dir'].exists():
        result_files = list(PATHS['results_dir'].glob('*.png'))
        status['results'] = len(result_files)
        print(f"   📊 예측 결과: {status['results']}개")
    
    return status

def show_main_menu():
    """메인 메뉴 표시"""
    print(f"\n📋 메인 메뉴")
    print("=" * 50)
    print(f"1. 🏷️  이미지 라벨링")
    print(f"2. 🚀 모델 훈련")
    print(f"3. 🔍 예측 및 테스트")
    print(f"4. 🛠️  시스템 관리")
    print(f"5. 📚 도움말")
    print(f"6. 🚪 종료")

def run_labeling():
    """라벨링 도구 실행"""
    print(f"\n🏷️ 이미지 라벨링 도구 실행")
    print("-" * 30)
    
    try:
        subprocess.run([sys.executable, 'run_labeling.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 라벨링 도구 실행 실패: {e}")
    except FileNotFoundError:
        print(f"❌ run_labeling.py 파일을 찾을 수 없습니다.")
        print(f"라벨링 도구를 직접 실행하세요:")
        print(f"   python labeling_tool.py")

def run_training():
    """모델 훈련 실행"""
    print(f"\n🚀 YOLO 모델 훈련 실행")
    print("-" * 30)
    
    try:
        subprocess.run([sys.executable, 'run_training.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 모델 훈련 실행 실패: {e}")
    except FileNotFoundError:
        print(f"❌ run_training.py 파일을 찾을 수 없습니다.")
        print(f"훈련을 직접 실행하세요:")
        print(f"   python train_yolo.py")

def run_prediction():
    """예측 시스템 실행"""
    print(f"\n🔍 예측 시스템 실행")
    print("-" * 30)
    
    try:
        subprocess.run([sys.executable, 'run_prediction.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 예측 시스템 실행 실패: {e}")
    except FileNotFoundError:
        print(f"❌ run_prediction.py 파일을 찾을 수 없습니다.")
        print(f"예측을 직접 실행하세요:")
        print(f"   python predict_yolo.py")

def system_management():
    """시스템 관리 메뉴"""
    while True:
        print(f"\n🛠️ 시스템 관리")
        print("=" * 30)
        print(f"1. 📊 상세 통계 보기")
        print(f"2. 🗂️  폴더 열기")
        print(f"3. 🧹 결과 파일 정리")
        print(f"4. ⚙️  설정 확인")
        print(f"5. 🔄 시스템 재시작")
        print(f"6. ⬅️  메인 메뉴로")
        
        choice = input(f"\n선택 (1-6): ").strip()
        
        if choice == '1':
            show_detailed_stats()
        elif choice == '2':
            open_folders()
        elif choice == '3':
            cleanup_results()
        elif choice == '4':
            show_config()
        elif choice == '5':
            restart_system()
        elif choice == '6':
            break
        else:
            print("잘못된 선택입니다.")

def show_detailed_stats():
    """상세 통계 표시"""
    print(f"\n📊 상세 시스템 통계")
    print("=" * 40)
    
    # 라벨링 통계
    if PATHS['images_dir'].exists():
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(PATHS['images_dir'].glob(ext))
        
        class_counts = {cls: 0 for cls in CLASSES}
        total_objects = 0
        
        for img_file in image_files:
            json_file = img_file.with_suffix('.json')
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for ann in data.get('annotations', []):
                            class_name = ann.get('class_name', '')
                            if class_name in class_counts:
                                class_counts[class_name] += 1
                                total_objects += 1
                except:
                    continue
        
        print(f"📝 라벨링 통계:")
        print(f"   - 총 객체 수: {total_objects}개")
        for class_name, count in class_counts.items():
            print(f"   - {class_name}: {count}개")
    
    # 모델 통계
    if PATHS['models_dir'].exists():
        model_files = list(PATHS['models_dir'].glob('*.pth'))
        print(f"\n🧠 모델 통계:")
        print(f"   - 모델 개수: {len(model_files)}개")
        
        for model_file in model_files:
            try:
                import torch
                checkpoint = torch.load(model_file, map_location='cpu')
                epoch = checkpoint.get('epoch', 'Unknown')
                loss = checkpoint.get('loss', 'Unknown')
                print(f"   - {model_file.name}: 에포크 {epoch}, 손실 {loss:.4f}")
            except:
                print(f"   - {model_file.name}: 정보 읽기 실패")
    
    # 디스크 사용량
    try:
        import shutil
        total, used, free = shutil.disk_usage(PATHS['base_dir'])
        
        def format_bytes(bytes_value):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_value < 1024:
                    return f"{bytes_value:.1f}{unit}"
                bytes_value /= 1024
            return f"{bytes_value:.1f}TB"
        
        print(f"\n💾 디스크 사용량:")
        print(f"   - 전체: {format_bytes(total)}")
        print(f"   - 사용: {format_bytes(used)}")
        print(f"   - 여유: {format_bytes(free)}")
    except:
        pass

def open_folders():
    """폴더 열기"""
    print(f"\n🗂️ 폴더 열기")
    print("=" * 20)
    print(f"1. 이미지 폴더")
    print(f"2. 모델 폴더")
    print(f"3. 결과 폴더")
    print(f"4. 전체 프로젝트 폴더")
    
    choice = input(f"\n선택 (1-4): ").strip()
    
    import subprocess
    import platform
    
    folders = {
        '1': PATHS['images_dir'],
        '2': PATHS['models_dir'],
        '3': PATHS['results_dir'],
        '4': PATHS['base_dir']
    }
    
    if choice in folders:
        folder_path = folders[choice]
        try:
            if platform.system() == 'Windows':
                subprocess.run(['explorer', str(folder_path)])
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', str(folder_path)])
            else:  # Linux
                subprocess.run(['xdg-open', str(folder_path)])
            print(f"✅ 폴더 열기: {folder_path}")
        except Exception as e:
            print(f"❌ 폴더 열기 실패: {e}")
            print(f"수동으로 열어주세요: {folder_path}")
    else:
        print("잘못된 선택입니다.")

def cleanup_results():
    """결과 파일 정리"""
    print(f"\n🧹 결과 파일 정리")
    print("=" * 20)
    
    if not PATHS['results_dir'].exists():
        print("정리할 결과 파일이 없습니다.")
        return
    
    result_files = list(PATHS['results_dir'].glob('*'))
    
    if not result_files:
        print("정리할 파일이 없습니다.")
        return
    
    print(f"정리 대상: {len(result_files)}개 파일")
    for file_path in result_files[:10]:  # 처음 10개만 표시
        print(f"   - {file_path.name}")
    
    if len(result_files) > 10:
        print(f"   ... 외 {len(result_files)-10}개")
    
    confirm = input(f"\n정말로 삭제하시겠습니까? (y/N): ")
    if confirm.lower() in ['y', 'yes']:
        try:
            import shutil
            shutil.rmtree(PATHS['results_dir'])
            PATHS['results_dir'].mkdir()
            print(f"✅ 결과 파일 정리 완료")
        except Exception as e:
            print(f"❌ 정리 실패: {e}")
    else:
        print("정리가 취소되었습니다.")

def show_config():
    """설정 정보 표시"""
    print(f"\n⚙️ 현재 설정")
    print("=" * 30)
    
    print(f"🎯 클래스:")
    for i, cls in enumerate(CLASSES, 1):
        print(f"   {i}. {cls}")
    
    print(f"\n📂 경로:")
    for name, path in PATHS.items():
        print(f"   - {name}: {path}")
    
    try:
        from config import TRAIN_CONFIG, PREDICT_CONFIG, MODEL_CONFIG
        
        print(f"\n🚀 훈련 설정:")
        for key, value in TRAIN_CONFIG.items():
            print(f"   - {key}: {value}")
        
        print(f"\n🔍 예측 설정:")
        for key, value in PREDICT_CONFIG.items():
            print(f"   - {key}: {value}")
        
        print(f"\n🧠 모델 설정:")
        for key, value in MODEL_CONFIG.items():
            print(f"   - {key}: {value}")
    except:
        pass

def restart_system():
    """시스템 재시작"""
    print(f"\n🔄 시스템 재시작")
    print("=" * 20)
    
    confirm = input(f"시스템을 재시작하시겠습니까? (y/N): ")
    if confirm.lower() in ['y', 'yes']:
        print("시스템을 재시작합니다...")
        time.sleep(1)
        subprocess.run([sys.executable, __file__])
        sys.exit(0)

def main():
    """메인 실행 함수"""
    show_banner()
    
    while True:
        # 시스템 상태 확인
        status = check_system_status()
        
        # 메인 메뉴 표시
        show_main_menu()
        
        try:
            choice = input(f"\n선택하세요 (1-6): ").strip()
            
            if choice == '1':
                run_labeling()
            elif choice == '2':
                if status['labeled_images'] == 0:
                    print(f"\n⚠️ 라벨링된 이미지가 없습니다!")
                    print(f"먼저 이미지 라벨링을 완료하세요.")
                else:
                    run_training()
            elif choice == '3':
                if status['models'] == 0:
                    print(f"\n⚠️ 훈련된 모델이 없습니다!")
                    print(f"먼저 모델 훈련을 완료하세요.")
                else:
                    run_prediction()
            elif choice == '4':
                system_management()
            elif choice == '5':
                show_help()
            elif choice == '6':
                print(f"\n🚪 YOLO 시스템을 종료합니다. 안녕히가세요!")
                break
            else:
                print(f"❌ 잘못된 선택입니다. 1-6 중에서 선택하세요.")
        
        except KeyboardInterrupt:
            print(f"\n\n🚪 시스템이 중단되었습니다. 안녕히가세요!")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {str(e)}")
            time.sleep(2)

if __name__ == "__main__":
    main()