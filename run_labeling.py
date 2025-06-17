
import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from config import setup_directories, CLASSES, PATHS
    from labeling_tool import ImageLabelingTool
    import json
except ImportError as e:
    print(f"❌ 모듈 import 오류: {e}")
    print("필요한 파일들이 같은 폴더에 있는지 확인하세요:")
    print("  - config.py")
    print("  - labeling_tool.py")
    print("  - yolo_model.py")
    sys.exit(1)

def check_requirements():
    """필수 라이브러리 확인"""
    required_packages = ['tkinter', 'PIL', 'json', 'pathlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'PIL':
                from PIL import Image
            elif package == 'json':
                import json
            elif package == 'pathlib':
                from pathlib import Path
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 누락된 패키지: {', '.join(missing_packages)}")
        if 'tkinter' in missing_packages:
            print("Tkinter 설치:")
            print("  Ubuntu/Debian: sudo apt-get install python3-tk")
            print("  CentOS/RHEL: sudo yum install tkinter")
            print("  macOS: 기본 설치됨")
            print("  Windows: 기본 설치됨")
        return False
    
    return True

def show_labeling_stats():
    """라벨링 통계 표시"""
    images_dir = PATHS['images_dir']
    
    if not images_dir.exists():
        print(f"📁 이미지 폴더가 없습니다: {images_dir}")
        return
    
    # 이미지 파일 찾기
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(images_dir.glob(ext))
        image_files.extend(images_dir.glob(ext.upper()))
    
    # 라벨링된 파일 찾기
    labeled_files = []
    for img_file in image_files:
        json_file = img_file.with_suffix('.json')
        if json_file.exists():
            labeled_files.append((img_file, json_file))
    
    print(f"📊 라벨링 현황:")
    print(f"   - 전체 이미지: {len(image_files)}개")
    print(f"   - 라벨링 완료: {len(labeled_files)}개")
    
    if len(image_files) > 0:
        progress = len(labeled_files) / len(image_files) * 100
        print(f"   - 진행률: {progress:.1f}%")
    
    # 클래스별 통계
    if labeled_files:
        class_counts = {cls: 0 for cls in CLASSES}
        total_objects = 0
        
        for img_file, json_file in labeled_files:
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
        
        print(f"   - 총 객체 수: {total_objects}개")
        print(f"   - 클래스별 분포:")
        for class_name, count in class_counts.items():
            print(f"     • {class_name}: {count}개")

def create_sample_images_info():
    """샘플 이미지 정보 생성"""
    info_file = PATHS['images_dir'] / 'README.txt'
    
    
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(info_content.strip())

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🏷️ YOLO 이미지 라벨링 도구")
    print("=" * 60)
    
    # 필수 라이브러리 확인
    if not check_requirements():
        return
    
    # 디렉토리 설정
    print("📁 디렉토리 설정 중...")
    setup_directories()
    create_sample_images_info()
    
    # 현재 상태 표시
    show_labeling_stats()
    
    print(f"\n🎯 라벨링 클래스: {', '.join(CLASSES)}")
    print(f"📂 이미지 폴더: {PATHS['images_dir']}")
    
    # 이미지 폴더 확인
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(PATHS['images_dir'].glob(ext))
        image_files.extend(PATHS['images_dir'].glob(ext.upper()))
    
    if not image_files:
        print(f"\n⚠️ 이미지 폴더가 비어있습니다!")
        print(f"다음 폴더에 이미지를 넣어주세요: {PATHS['images_dir']}")
        print("그 다음 다시 실행하세요.")
        
        # 폴더 열기 (운영체제별)
        try:
            import subprocess
            import platform
            
            if platform.system() == 'Windows':
                subprocess.run(['explorer', str(PATHS['images_dir'])])
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', str(PATHS['images_dir'])])
            else:  # Linux
                subprocess.run(['xdg-open', str(PATHS['images_dir'])])
        except:
            pass
        
        return
    
    print(f"\n✅ {len(image_files)}개의 이미지를 찾았습니다.")
    
    # 사용법 안내
    print(f"\n📋 사용법:")
    print(f"1. 📁 '이미지 폴더 선택' → {PATHS['images_dir']} 선택")
    print(f"2. 🖼️ '이미지 로드' → 첫 번째 이미지 로드")
    print(f"3. 🎯 클래스 선택 (숫자키 1-{len(CLASSES)} 또는 라디오 버튼)")
    print(f"4. 🖱️ 마우스 드래그로 바운딩 박스 그리기")
    print(f"5. 💾 '저장' (S키) → 다음 이미지로 자동 이동")
    
    # 실행 확인
    response = input(f"\n라벨링 도구를 실행하시겠습니까? (Y/n): ")
    if response.lower() in ['', 'y', 'yes']:
        print("\n🚀 라벨링 도구 실행 중...")
        
        try:
            # 라벨링 도구 실행
            tool = ImageLabelingTool()
            tool.run()
            
            print("\n🏷️ 라벨링 도구가 종료되었습니다.")
            
            # 완료 후 통계 표시
            print("\n📊 라벨링 완료 후 통계:")
            show_labeling_stats()
            
        except Exception as e:
            print(f"\n❌ 라벨링 도구 실행 오류: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("라벨링이 취소되었습니다.")

if __name__ == "__main__":
    main()