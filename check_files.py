# check_files.py - 파일 위치 확인
from pathlib import Path

def check_files():
    images_dir = Path("yolo_dataset/images")
    annotations_dir = Path("yolo_dataset/annotations")
    
    print("📁 파일 위치 확인:")
    
    # 이미지 파일 확인
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"   이미지 파일: {len(image_files)}개")
    
    # images 폴더의 JSON 파일 확인
    json_in_images = list(images_dir.glob("*.json"))
    print(f"   images 폴더의 JSON: {len(json_in_images)}개")
    
    # annotations 폴더의 JSON 파일 확인
    json_in_annotations = list(annotations_dir.glob("*.json"))
    print(f"   annotations 폴더의 JSON: {len(json_in_annotations)}개")
    
    # 매칭 확인
    matched = 0
    for img_file in image_files[:5]:  # 처음 5개만 확인
        json_file = annotations_dir / f"{img_file.stem}.json"
        if json_file.exists():
            matched += 1
            print(f"   ✅ {img_file.name} → {json_file.name}")
        else:
            print(f"   ❌ {img_file.name} → JSON 없음")
    
    print(f"\n📊 요약:")
    print(f"   - 총 이미지: {len(image_files)}개")
    print(f"   - annotations 폴더 JSON: {len(json_in_annotations)}개")
    print(f"   - 매칭 확인 (처음 5개): {matched}/5")

if __name__ == "__main__":
    check_files()