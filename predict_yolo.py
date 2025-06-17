# predict_yolo.py - YOLO 예측 및 테스트

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import time
import json
from pathlib import Path
import numpy as np

from yolo_model import SimpleYOLO, decode_predictions, non_max_suppression, visualize_predictions, load_model
from config import *

class YOLOPredictor:
    """YOLO 예측 클래스"""
    
    def __init__(self, model_path, confidence_threshold=0.3, nms_threshold=0.5):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = DEVICE
        
        # 모델 로드
        self.load_model()
        
        # 전처리 변환
        self.transform = transforms.Compose([
            transforms.Resize((MODEL_CONFIG['img_size'], MODEL_CONFIG['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
        
    def load_model(self):
        """훈련된 모델 로드"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {self.model_path}")
        
        print(f"🔄 모델 로딩 중: {self.model_path}")
        
        self.model = load_model(self.model_path, self.device)
        
        print("✅ 모델 로드 완료!")
        
    def predict_image(self, image_path, save_result=True, show_result=True):
        """단일 이미지 예측"""
        print(f"\n🔍 이미지 예측: {image_path}")
        
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        
        print(f"   - 원본 크기: {orig_width}×{orig_height}")
        
        # 전처리
        start_time = time.time()
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 예측
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # 후처리
        detections = decode_predictions(output[0], self.confidence_threshold)
        
        # NMS 적용
        if detections:
            boxes = torch.tensor([[d[0], d[1], d[2], d[3]] for d in detections])
            scores = torch.tensor([d[4] for d in detections])
            
            if len(boxes) > 1:
                final_boxes, final_scores = non_max_suppression(
                    boxes, scores, self.nms_threshold
                )
                
                # 최종 검출 결과 생성
                final_detections = []
                for i in range(len(final_boxes)):
                    x, y, w, h = final_boxes[i].tolist()
                    conf = final_scores[i].item()
                    class_id = detections[i][5]  # 원래 클래스 ID 유지
                    
                    # 원본 크기로 스케일링
                    x_pixel = x * orig_width
                    y_pixel = y * orig_height
                    w_pixel = w * orig_width
                    h_pixel = h * orig_height
                    
                    final_detections.append({
                        'class': CLASSES[class_id],
                        'confidence': conf,
                        'bbox': [x_pixel, y_pixel, w_pixel, h_pixel],
                        'class_id': class_id
                    })
            else:
                # 단일 검출
                if detections:
                    d = detections[0]
                    x, y, w, h, conf, class_id = d
                    
                    x_pixel = x * orig_width
                    y_pixel = y * orig_height
                    w_pixel = w * orig_width
                    h_pixel = h * orig_height
                    
                    final_detections = [{
                        'class': CLASSES[class_id],
                        'confidence': conf,
                        'bbox': [x_pixel, y_pixel, w_pixel, h_pixel],
                        'class_id': class_id
                    }]
                else:
                    final_detections = []
        else:
            final_detections = []
        
        inference_time = time.time() - start_time
        
        # 결과 출력
        print(f"   - 추론 시간: {inference_time*1000:.1f}ms")
        print(f"   - 검출된 객체: {len(final_detections)}개")
        
        for i, det in enumerate(final_detections):
            x, y, w, h = det['bbox']
            print(f"     {i+1}. {det['class']}: {det['confidence']:.3f} "
                  f"({int(x)}, {int(y)}, {int(w)}, {int(h)})")
        
        # 결과 시각화 및 저장
        if save_result or show_result:
            result_path = None
            if save_result:
                PATHS['results_dir'].mkdir(parents=True, exist_ok=True)
                result_filename = f"result_{Path(image_path).stem}.png"
                result_path = PATHS['results_dir'] / result_filename
            
            visualize_predictions(
                image, final_detections, 
                save_path=result_path, 
                show_plot=show_result
            )
        
        return final_detections, inference_time
    
    def predict_batch(self, image_folder, output_file=None):
        """폴더 내 모든 이미지 예측"""
        image_folder = Path(image_folder)
        
        if not image_folder.exists():
            raise FileNotFoundError(f"폴더가 없습니다: {image_folder}")
        
        # 이미지 파일 찾기
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(image_folder.glob(ext))
            image_files.extend(image_folder.glob(ext.upper()))
        
        if not image_files:
            print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_folder}")
            return
        
        print(f"\n📁 배치 예측: {len(image_files)}개 이미지")
        print("=" * 50)
        
        all_results = []
        total_time = 0
        total_objects = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] {image_path.name}")
            
            try:
                detections, inference_time = self.predict_image(
                    image_path, save_result=True, show_result=False
                )
                
                total_time += inference_time
                total_objects += len(detections)
                
                # 결과 기록
                result = {
                    'image_path': str(image_path),
                    'detections': detections,
                    'inference_time': inference_time,
                    'num_objects': len(detections)
                }
                all_results.append(result)
                
            except Exception as e:
                print(f"   ❌ 예측 실패: {str(e)}")
                continue
        
        # 전체 통계
        avg_time = total_time / len(all_results) if all_results else 0
        avg_objects = total_objects / len(all_results) if all_results else 0
        
        print("\n" + "=" * 50)
        print("📊 배치 예측 완료!")
        print(f"   - 처리된 이미지: {len(all_results)}개")
        print(f"   - 총 검출 객체: {total_objects}개")
        print(f"   - 평균 추론 시간: {avg_time*1000:.1f}ms")
        print(f"   - 이미지당 평균 객체: {avg_objects:.1f}개")
        print(f"   - 결과 저장 위치: {PATHS['results_dir']}")
        
        # 결과 파일 저장
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = PATHS['results_dir'] / 'batch_results.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'model_path': str(self.model_path),
                    'confidence_threshold': self.confidence_threshold,
                    'nms_threshold': self.nms_threshold,
                    'classes': CLASSES
                },
                'statistics': {
                    'total_images': len(all_results),
                    'total_objects': total_objects,
                    'avg_inference_time': avg_time,
                    'avg_objects_per_image': avg_objects
                },
                'results': all_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"   - 결과 JSON: {output_path}")
        
        return all_results
    
    def evaluate_performance(self, test_folder=None):
        """성능 평가 (간단한 통계)"""
        if test_folder:
            results = self.predict_batch(test_folder, show_result=False)
        else:
            print("테스트 폴더가 지정되지 않았습니다.")
            return
        
        if not results:
            return
        
        # 클래스별 통계
        class_counts = {cls: 0 for cls in CLASSES}
        confidence_scores = {cls: [] for cls in CLASSES}
        
        for result in results:
            for det in result['detections']:
                class_name = det['class']
                confidence = det['confidence']
                
                class_counts[class_name] += 1
                confidence_scores[class_name].append(confidence)
        
        # 통계 출력
        print("\n📊 성능 통계:")
        print("=" * 40)
        
        for class_name in CLASSES:
            count = class_counts[class_name]
            if count > 0:
                avg_conf = np.mean(confidence_scores[class_name])
                min_conf = np.min(confidence_scores[class_name])
                max_conf = np.max(confidence_scores[class_name])
                
                print(f"{class_name}:")
                print(f"   - 검출 수: {count}")
                print(f"   - 평균 신뢰도: {avg_conf:.3f}")
                print(f"   - 신뢰도 범위: {min_conf:.3f} ~ {max_conf:.3f}")
            else:
                print(f"{class_name}: 검출되지 않음")
        
        # 신뢰도 분포 시각화
        self.plot_confidence_distribution(confidence_scores)
    
    def plot_confidence_distribution(self, confidence_scores):
        """신뢰도 분포 시각화"""
        plt.figure(figsize=(12, 6))
        
        # 클래스별 신뢰도 분포
        plt.subplot(1, 2, 1)
        for i, (class_name, scores) in enumerate(confidence_scores.items()):
            if scores:
                plt.hist(scores, bins=20, alpha=0.7, label=class_name, color=COLORS[i])
        
        plt.xlabel('신뢰도')
        plt.ylabel('빈도')
        plt.title('클래스별 신뢰도 분포')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 전체 신뢰도 분포
        plt.subplot(1, 2, 2)
        all_scores = []
        for scores in confidence_scores.values():
            all_scores.extend(scores)
        
        if all_scores:
            plt.hist(all_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(np.mean(all_scores), color='red', linestyle='--', 
                       label=f'평균: {np.mean(all_scores):.3f}')
            plt.axvline(self.confidence_threshold, color='green', linestyle='--',
                       label=f'임계값: {self.confidence_threshold}')
        
        plt.xlabel('신뢰도')
        plt.ylabel('빈도')
        plt.title('전체 신뢰도 분포')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        plot_path = PATHS['results_dir'] / 'confidence_distribution.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 신뢰도 분포 그래프: {plot_path}")

def check_model_file(model_path):
    """모델 파일 확인"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        
        # 대안 모델 찾기
        models_dir = PATHS['models_dir']
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pth'))
            if model_files:
                print(f"\n📁 사용 가능한 모델들:")
                for i, model_file in enumerate(model_files, 1):
                    print(f"   {i}. {model_file.name}")
                return None
        
        print("\n💡 먼저 train_yolo.py를 실행해서 모델을 훈련하세요.")
        return None
    
    return model_path

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='YOLO 모델 예측')
    parser.add_argument('--model', type=str, 
                       default='yolo_dataset/models/best_yolo_model.pth',
                       help='모델 파일 경로')
    parser.add_argument('--image', type=str,
                       help='예측할 이미지 파일 경로')
    parser.add_argument('--folder', type=str,
                       help='예측할 이미지 폴더 경로')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='신뢰도 임계값')
    parser.add_argument('--nms', type=float, default=0.5,
                       help='NMS IoU 임계값')
    parser.add_argument('--no-show', action='store_true',
                       help='결과 이미지 표시 안함')
    parser.add_argument('--no-save', action='store_true',
                       help='결과 이미지 저장 안함')
    parser.add_argument('--eval', action='store_true',
                       help='성능 평가 수행')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 YOLO 객체 검출 예측 시스템")
    print("=" * 60)
    print(f"클래스: {', '.join(CLASSES)}")
    
    # 디렉토리 설정
    setup_directories()
    
    # 모델 파일 확인
    model_path = check_model_file(args.model)
    if not model_path:
        return
    
    try:
        # 예측기 초기화
        predictor = YOLOPredictor(
            model_path=model_path,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms
        )
        
        print(f"\n🛠️ 예측 설정:")
        print(f"   - 신뢰도 임계값: {args.confidence}")
        print(f"   - NMS 임계값: {args.nms}")
        
        # 예측 실행
        if args.image:
            # 단일 이미지 예측
            if not Path(args.image).exists():
                print(f"❌ 이미지 파일이 없습니다: {args.image}")
                return
            
            predictor.predict_image(
                args.image,
                save_result=not args.no_save,
                show_result=not args.no_show
            )
            
        elif args.folder:
            # 폴더 배치 예측
            if not Path(args.folder).exists():
                print(f"❌ 폴더가 없습니다: {args.folder}")
                return
            
            predictor.predict_batch(args.folder)
            
            # 성능 평가
            if args.eval:
                predictor.evaluate_performance(args.folder)
                
        else:
            # 대화형 모드
            print("\n🎮 대화형 모드")
            print("이미지 파일 경로를 입력하거나 'quit'로 종료하세요.")
            
            while True:
                image_path = input("\n이미지 경로: ").strip()
                
                if image_path.lower() in ['quit', 'q', 'exit']:
                    break
                
                if not image_path:
                    continue
                
                if not Path(image_path).exists():
                    print(f"❌ 파일이 없습니다: {image_path}")
                    continue
                
                try:
                    predictor.predict_image(
                        image_path,
                        save_result=not args.no_save,
                        show_result=not args.no_show
                    )
                except Exception as e:
                    print(f"❌ 예측 실패: {str(e)}")
    
    except Exception as e:
        print(f"\n❌ 예측 시스템 오류: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()