# fix_class_ids.py
import json
from pathlib import Path

def fix_class_ids():
    images_dir = Path("yolo_dataset/images")
    
    # 클래스 매핑: 이전 인덱스 → 새 인덱스
    class_mapping = {
        0: 0,  # cat → cat (변경 없음)
        3: 1,  # person (이전 3번) → person (새 1번)
    }
    
    fixed_count = 0
    
    for json_file in images_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            changed = False
            for ann in data.get('annotations', []):
                old_id = ann.get('class_id')
                if old_id in class_mapping:
                    new_id = class_mapping[old_id]
                    ann['class_id'] = new_id
                    # 클래스명도 업데이트
                    if new_id == 0:
                        ann['class_name'] = 'cat'
                    elif new_id == 1:
                        ann['class_name'] = 'person'
                    changed = True
            
            if changed:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                fixed_count += 1
                
        except Exception as e:
            print(f"오류 {json_file.name}: {e}")
    
    print(f"✅ {fixed_count}개 파일 수정 완료!")

if __name__ == "__main__":
    fix_class_ids()