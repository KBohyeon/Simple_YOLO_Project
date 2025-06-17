# train_yolo.py - YOLO 모델 훈련

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
import argparse
from pathlib import Path

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

from yolo_model import SimpleYOLO, YOLOLoss, RealImageDataset, get_transform, save_model
from config import *

class YOLOTrainer:
    """YOLO 훈련 클래스"""
    
    def __init__(self, image_folder, config=None):
        self.image_folder = Path(image_folder)
        self.config = config or TRAIN_CONFIG
        self.device = DEVICE
        
        # 모델 및 데이터 초기화
        self.setup_model()
        self.setup_data()
        
        # 훈련 기록
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def setup_model(self):
        """모델 및 최적화 설정"""
        print("🧠 모델 초기화 중...")
        
        # 모델
        self.model = SimpleYOLO(
            num_classes=MODEL_CONFIG['num_classes'],
            grid_size=MODEL_CONFIG['grid_size']
        ).to(self.device)
        
        # 손실 함수
        self.criterion = YOLOLoss(
            lambda_coord=self.config['lambda_coord'],
            lambda_noobj=self.config['lambda_noobj']
        )
        
        # 최적화기
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 스케줄러
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config['epochs']//3,
            gamma=0.1
        )
        
        # 모델 정보 출력
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   - 총 파라미터: {total_params:,}")
        print(f"   - 훈련 가능 파라미터: {trainable_params:,}")
        print(f"   - 디바이스: {self.device}")
        
    def setup_data(self):
        """데이터셋 및 데이터로더 설정"""
        print("📊 데이터셋 준비 중...")
        
        # 데이터 변환
        train_transform = get_transform(train=True)
        val_transform = get_transform(train=False)
        
        # 전체 데이터셋
        full_dataset = RealImageDataset(
            self.image_folder,
            grid_size=MODEL_CONFIG['grid_size'],
            img_size=MODEL_CONFIG['img_size'],
            transform=train_transform
        )
        
        if len(full_dataset) == 0:
            raise ValueError(f"❌ 라벨링된 이미지가 없습니다: {self.image_folder}")
        
        # 훈련/검증 분할
        train_size = int(self.config['train_val_split'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        # 검증 데이터셋에 다른 변환 적용
        val_dataset.dataset = RealImageDataset(
            self.image_folder,
            grid_size=MODEL_CONFIG['grid_size'],
            img_size=MODEL_CONFIG['img_size'],
            transform=val_transform
        )
        
        # 데이터로더
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"   - 총 데이터: {len(full_dataset)}개")
        print(f"   - 훈련 데이터: {len(self.train_dataset)}개")
        print(f"   - 검증 데이터: {len(val_dataset)}개")
        print(f"   - 배치 크기: {self.config['batch_size']}")
        
    def train_epoch(self, epoch):
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        print(f"\n에포크 {epoch+1}/{self.config['epochs']} 훈련 중...")
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # 순전파
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑 (선택적)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 진행 상황 출력
            if batch_idx % max(1, num_batches // 10) == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"   배치 {batch_idx+1}/{num_batches} ({progress:.1f}%) - "
                      f"손실: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        print(f"에포크 {epoch+1} 검증 중...")
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """체크포인트 저장"""
        # 모델 디렉토리 생성
        PATHS['models_dir'].mkdir(parents=True, exist_ok=True)
        
        if is_best:
            save_path = PATHS['models_dir'] / 'best_yolo_model.pth'
            save_model(self.model, self.optimizer, epoch, val_loss, save_path)
            print(f"✅ 최고 모델 저장: {save_path}")
        
        # 마지막 모델도 저장
        last_save_path = PATHS['models_dir'] / 'last_yolo_model.pth'
        save_model(self.model, self.optimizer, epoch, val_loss, last_save_path)
    
    def plot_training_history(self):
        """훈련 과정 시각화"""
        plt.figure(figsize=(12, 5))
        
        # 손실 그래프
        plt.subplot(1, 2, 1)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='훈련 손실', linewidth=2)
        if self.val_losses and max(self.val_losses) > 0:
            plt.plot(epochs, self.val_losses, 'r-', label='검증 손실', linewidth=2)
        plt.xlabel('에포크')
        plt.ylabel('손실')
        plt.title('훈련 과정')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 학습률 그래프
        plt.subplot(1, 2, 2)
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        plt.plot(epochs, [lrs[0]] * len(epochs), 'g-', linewidth=2)
        plt.xlabel('에포크')
        plt.ylabel('학습률')
        plt.title('학습률 변화')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        
        # 저장
        plot_path = PATHS['models_dir'] / 'training_history.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 훈련 그래프 저장: {plot_path}")
    
    def train(self):
        """전체 훈련 과정"""
        print("=" * 70)
        print("🚀 YOLO 모델 훈련 시작!")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config['epochs']):
                print(f"\n{'='*50}")
                
                # 훈련
                train_loss = self.train_epoch(epoch)
                
                # 검증
                val_loss = self.validate_epoch(epoch)
                
                # 스케줄러 업데이트
                self.scheduler.step()
                
                # 결과 출력
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\n에포크 {epoch+1} 완료:")
                print(f"   - 훈련 손실: {train_loss:.4f}")
                print(f"   - 검증 손실: {val_loss:.4f}")
                print(f"   - 학습률: {current_lr:.6f}")
                
                # 최고 모델 저장
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                self.save_checkpoint(epoch, val_loss, is_best)
                
                # 조기 종료 조건 (선택적)
                if train_loss < 0.01:
                    print(f"\n🎯 훈련 손실이 충분히 낮아졌습니다 ({train_loss:.4f})")
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️ 훈련이 중단되었습니다.")
        
        except Exception as e:
            print(f"\n❌ 훈련 중 오류 발생: {str(e)}")
            raise
        
        finally:
            # 훈련 완료
            end_time = time.time()
            training_time = end_time - start_time
            
            print("\n" + "=" * 70)
            print("🎉 훈련 완료!")
            print("=" * 70)
            print(f"⏱️ 총 훈련 시간: {training_time//60:.0f}분 {training_time%60:.1f}초")
            print(f"🏆 최고 검증 손실: {self.best_val_loss:.4f}")
            print(f"📁 모델 저장 위치: {PATHS['models_dir']}")
            
            # 훈련 그래프 생성
            self.plot_training_history()
            
            # 훈련 통계 저장
            self.save_training_stats(training_time)
    
    def save_training_stats(self, training_time):
        """훈련 통계 저장"""
        stats = {
            'config': self.config,
            'model_config': MODEL_CONFIG,
            'training_time': training_time,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'epochs_completed': len(self.train_losses),
            'dataset_size': len(self.train_dataset) + len(self.val_loader.dataset),
            'classes': CLASSES
        }
        
        import json
        stats_path = PATHS['models_dir'] / 'training_stats.json'
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"📊 훈련 통계 저장: {stats_path}")

def check_dataset(image_folder):
    """데이터셋 상태 확인"""
    from config import PATHS, CLASSES
    
    image_folder = Path(image_folder)
    annotations_dir = PATHS['annotations_dir']  # ← 핵심 추가!
    
    if not image_folder.exists():
        print(f"❌ 폴더가 존재하지 않습니다: {image_folder}")
        return False
    
    # 이미지 파일 찾기
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(image_folder.glob(ext))
        image_files.extend(image_folder.glob(ext.upper()))
    
    # 라벨링된 이미지 찾기 (annotations 폴더에서!)
    labeled_files = []
    for img_file in image_files:
        annotation_path = annotations_dir / f"{img_file.stem}.json"  # ← 핵심 수정!
        if annotation_path.exists():
            labeled_files.append((img_file, annotation_path))
    
    print(f"📊 데이터셋 현황:")
    print(f"   - 전체 이미지: {len(image_files)}개")
    print(f"   - 라벨링된 이미지: {len(labeled_files)}개")
    
    if len(labeled_files) == 0:
        print("❌ 라벨링된 이미지가 없습니다!")
        print("먼저 labeling_tool.py를 실행해서 이미지들을 라벨링하세요.")
        return False
    
    # 클래스별 통계
    class_counts = {cls: 0 for cls in CLASSES}
    total_objects = 0
    
    for img_file, json_file in labeled_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:  # ← 인코딩 추가!
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
    
    # 권장사항
    min_per_class = 20
    insufficient_classes = [cls for cls, count in class_counts.items() if count < min_per_class]
    
    if insufficient_classes:
        print(f"\n⚠️ 권장사항: 다음 클래스들의 데이터가 부족합니다 (최소 {min_per_class}개 권장):")
        for cls in insufficient_classes:
            print(f"   - {cls}: {class_counts[cls]}개")
    
    return len(labeled_files) > 0

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='YOLO 모델 훈련')
    parser.add_argument('--data', type=str, default='yolo_dataset/images',
                       help='이미지 폴더 경로')
    parser.add_argument('--epochs', type=int, default=30,
                       help='훈련 에포크 수')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='학습률')
    parser.add_argument('--check-only', action='store_true',
                       help='데이터셋 확인만 수행')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 YOLO 모델 훈련 시스템")
    print("=" * 70)
    print(f"클래스: {', '.join(CLASSES)}")
    print(f"데이터 폴더: {args.data}")
    
    # 디렉토리 설정
    setup_directories()
    
    # 데이터셋 확인
    if not check_dataset(args.data):
        return
    
    if args.check_only:
        print("\n✅ 데이터셋 확인 완료!")
        return
    
    # 훈련 설정 업데이트
    config = TRAIN_CONFIG.copy()
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    
    print(f"\n🛠️ 훈련 설정:")
    for key, value in config.items():
        print(f"   - {key}: {value}")
    
    # 사용자 확인
    response = input("\n훈련을 시작하시겠습니까? (y/N): ")
    if response.lower() != 'y':
        print("훈련이 취소되었습니다.")
        return
    
    try:
        # 훈련 시작
        trainer = YOLOTrainer(args.data, config)
        trainer.train()
        
    except Exception as e:
        print(f"\n❌ 훈련 실패: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()