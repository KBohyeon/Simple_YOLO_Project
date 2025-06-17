

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from pathlib import Path
from config import *

class SimpleYOLO(nn.Module):
    """YOLO 모델"""
    def __init__(self, num_classes=NUM_CLASSES, grid_size=7):
        super(SimpleYOLO, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        
        # CNN 백본 네트워크
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14
            
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 7x7
        )
        
        # YOLO 헤드
        self.output_size = 5 + num_classes  # x, y, w, h, conf + classes
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, grid_size * grid_size * self.output_size)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = x.view(batch_size, self.grid_size, self.grid_size, self.output_size)
        return x

class YOLOLoss(nn.Module):
    """YOLO 손실 함수"""
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        
        pred_boxes = predictions[:, :, :, :4]
        pred_conf = predictions[:, :, :, 4:5]
        pred_class = predictions[:, :, :, 5:]
        
        target_boxes = targets[:, :, :, :4]
        target_conf = targets[:, :, :, 4:5]
        target_class = targets[:, :, :, 5:]
        
        obj_mask = target_conf.squeeze(-1) > 0
        noobj_mask = target_conf.squeeze(-1) == 0
        
        # 좌표 손실
        if obj_mask.sum() > 0:
            coord_loss = self.lambda_coord * F.mse_loss(
                pred_boxes[obj_mask], target_boxes[obj_mask], reduction='sum'
            )
        else:
            coord_loss = torch.tensor(0.0, device=predictions.device)
        
        # 객체 confidence 손실
        if obj_mask.sum() > 0:
            obj_conf_loss = F.mse_loss(
                pred_conf[obj_mask], target_conf[obj_mask], reduction='sum'
            )
        else:
            obj_conf_loss = torch.tensor(0.0, device=predictions.device)
        
        # 객체 없는 셀 confidence 손실
        if noobj_mask.sum() > 0:
            noobj_conf_loss = self.lambda_noobj * F.mse_loss(
                pred_conf[noobj_mask], target_conf[noobj_mask], reduction='sum'
            )
        else:
            noobj_conf_loss = torch.tensor(0.0, device=predictions.device)
        
        # 클래스 손실
        if obj_mask.sum() > 0:
            class_loss = F.mse_loss(
                pred_class[obj_mask], target_class[obj_mask], reduction='sum'
            )
        else:
            class_loss = torch.tensor(0.0, device=predictions.device)
        
        total_loss = coord_loss + obj_conf_loss + noobj_conf_loss + class_loss
        return total_loss / batch_size

class RealImageDataset(Dataset):
    """실제 이미지 데이터셋"""
    def __init__(self, image_folder, grid_size=7, img_size=224, transform=None):
        self.image_folder = Path(image_folder)
        self.grid_size = grid_size
        self.img_size = img_size
        self.transform = transform
        
        # 이미지 파일과 어노테이션 파일 찾기
        self.data_pairs = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        # annotations 폴더 경로
        from config import PATHS
        annotations_dir = PATHS['annotations_dir']
        
        for ext in extensions:
            for img_path in self.image_folder.glob(ext):
                # annotations 폴더에서 JSON 파일 찾기
                annotation_path = annotations_dir / (img_path.stem + '.json')
                if annotation_path.exists():
                    self.data_pairs.append((img_path, annotation_path))
        
        print(f"총 {len(self.data_pairs)}개의 라벨링된 이미지를 찾았습니다.")
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        img_path, annotation_path = self.data_pairs[idx]
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # 어노테이션 로드
        with open(annotation_path, 'r', encoding='utf-8') as f:  # ← encoding 추가!
            data = json.load(f)
        
        annotations = data['annotations']
        
        # 이미지 크기 조정
        image = image.resize((self.img_size, self.img_size))
        
        # YOLO 타겟 생성
        target = self.create_yolo_target(annotations, orig_width, orig_height)
        
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
    def create_yolo_target(self, annotations, orig_width, orig_height):
        """어노테이션을 YOLO 타겟 형식으로 변환"""
        target = torch.zeros(self.grid_size, self.grid_size, 5 + NUM_CLASSES)
        
        for ann in annotations:
            class_id = ann['class_id']
            x1, y1, x2, y2 = ann['bbox']
            
            # 정규화된 좌표
            x_center = ((x1 + x2) / 2) / orig_width
            y_center = ((y1 + y2) / 2) / orig_height
            width = (x2 - x1) / orig_width
            height = (y2 - y1) / orig_height
            
            # 그리드 셀 찾기
            grid_x = int(x_center * self.grid_size)
            grid_y = int(y_center * self.grid_size)
            
            grid_x = min(grid_x, self.grid_size - 1)
            grid_y = min(grid_y, self.grid_size - 1)
            
            # 이미 객체가 있는 셀이면 스킵
            if target[grid_y, grid_x, 4] > 0:
                continue
            
            # 셀 내 상대 위치
            x_offset = (x_center * self.grid_size) - grid_x
            y_offset = (y_center * self.grid_size) - grid_y
            
            # 타겟 설정
            target[grid_y, grid_x, 0] = x_offset
            target[grid_y, grid_x, 1] = y_offset
            target[grid_y, grid_x, 2] = width
            target[grid_y, grid_x, 3] = height
            target[grid_y, grid_x, 4] = 1.0  # confidence
            target[grid_y, grid_x, 5 + class_id] = 1.0  # class
        
        return target

def get_transform(train=True):
    """데이터 변환 함수"""
    if train:
        # 훈련용 (데이터 증강 포함)
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    else:
        # 테스트용
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    
    return transform

def decode_predictions(prediction, confidence_threshold=0.3):
    """YOLO 출력을 바운딩 박스로 디코딩"""
    detections = []
    grid_size = prediction.size(0)
    
    for i in range(grid_size):
        for j in range(grid_size):
            cell_pred = prediction[i, j]
            
            x = cell_pred[0].item()
            y = cell_pred[1].item()
            w = cell_pred[2].item()
            h = cell_pred[3].item()
            confidence = torch.sigmoid(cell_pred[4]).item()
            
            if confidence > confidence_threshold:
                class_probs = torch.softmax(cell_pred[5:], dim=0)
                class_id = torch.argmax(class_probs).item()
                class_confidence = class_probs[class_id].item()
                
                final_confidence = confidence * class_confidence
                
                if final_confidence > confidence_threshold:
                    abs_x = (x + j) / grid_size
                    abs_y = (y + i) / grid_size
                    
                    detections.append([abs_x, abs_y, abs(w), abs(h), 
                                     final_confidence, class_id])
    
    return detections

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression"""
    if len(boxes) == 0:
        return [], []
    
    boxes = torch.tensor(boxes) if not isinstance(boxes, torch.Tensor) else boxes
    scores = torch.tensor(scores) if not isinstance(scores, torch.Tensor) else scores
    
    # 점수 순으로 정렬
    indices = torch.argsort(scores, descending=True)
    
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current.item())
        
        if len(indices) == 1:
            break
        
        # IoU 계산
        current_box = boxes[current].unsqueeze(0)
        other_boxes = boxes[indices[1:]]
        
        ious = calculate_iou(current_box, other_boxes)
        
        # IoU가 임계값 미만인 박스들만 유지
        indices = indices[1:][ious.squeeze() < iou_threshold]
    
    return boxes[keep], scores[keep]

def calculate_iou(box1, box2):
    """IoU (Intersection over Union) 계산"""
    # box format: (x_center, y_center, width, height)
    
    # 좌상단, 우하단 좌표로 변환
    box1_x1 = box1[:, 0] - box1[:, 2] / 2
    box1_y1 = box1[:, 1] - box1[:, 3] / 2
    box1_x2 = box1[:, 0] + box1[:, 2] / 2
    box1_y2 = box1[:, 1] + box1[:, 3] / 2
    
    box2_x1 = box2[:, 0] - box2[:, 2] / 2
    box2_y1 = box2[:, 1] - box2[:, 3] / 2
    box2_x2 = box2[:, 0] + box2[:, 2] / 2
    box2_y2 = box2[:, 1] + box2[:, 3] / 2
    
    # 교집합 영역
    inter_x1 = torch.max(box1_x1.unsqueeze(1), box2_x1.unsqueeze(0))
    inter_y1 = torch.max(box1_y1.unsqueeze(1), box2_y1.unsqueeze(0))
    inter_x2 = torch.min(box1_x2.unsqueeze(1), box2_x2.unsqueeze(0))
    inter_y2 = torch.min(box1_y2.unsqueeze(1), box2_y2.unsqueeze(0))
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 각 박스의 면적
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    # 합집합 면적
    union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter_area
    
    iou = inter_area / union_area
    return iou

def visualize_predictions(image, detections, save_path=None, show_plot=True):
    """예측 결과 시각화"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    for detection in detections:
        x, y, w, h = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class']
        class_id = detection['class_id']
        
        # 중심점 -> 좌상단 변환
        x1 = x - w/2
        y1 = y - h/2
        
        color = COLORS[class_id % len(COLORS)]
        
        rect = patches.Rectangle((x1, y1), w, h, 
                               linewidth=3, 
                               edgecolor=color, 
                               facecolor='none')
        ax.add_patch(rect)
        
        label = f'{class_name}: {confidence:.2f}'
        ax.text(x1, y1-10, label, 
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor=color, 
                         alpha=0.8),
                fontsize=12, color='white', weight='bold')
    
    ax.set_xlim(0, image.size[0])
    ax.set_ylim(image.size[1], 0)
    ax.axis('off')
    ax.set_title('YOLO 객체 검출 결과', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"결과 이미지 저장: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def load_model(model_path, device=DEVICE):
    """훈련된 모델 로드"""
    model = SimpleYOLO(num_classes=NUM_CLASSES).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 모델 로드 완료: {model_path}")
    print(f"   - 훈련 에포크: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   - 최종 손실: {checkpoint.get('loss', 'Unknown'):.4f}")
    
    return model

def save_model(model, optimizer, epoch, loss, save_path):
    """모델 저장"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'classes': CLASSES,
        'model_config': MODEL_CONFIG
    }, save_path)
    
    print(f"✅ 모델 저장: {save_path}")

if __name__ == "__main__":
    print("🧠 YOLO 모델 및 유틸리티 모듈")
    print(f"클래스: {CLASSES}")
    print(f"디바이스: {DEVICE}")
    
    # 테스트
    model = SimpleYOLO()
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")