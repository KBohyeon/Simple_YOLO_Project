

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw
import json
import os
from pathlib import Path
from config import CLASSES, COLORS, PATHS, setup_directories

class ImageLabelingTool:
    """이미지 라벨링 도구 GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLO 이미지 라벨링 도구")
        self.root.geometry("1400x900")
        
        # 상태 변수
        self.current_image = None
        self.current_image_path = None
        self.annotations = []
        self.current_class = 0
        self.drawing = False
        self.start_x = None
        self.start_y = None
        self.temp_rect = None
        self.image_files = []
        self.current_image_index = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """UI 구성"""
        # 메인 컨테이너
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 상단 툴바
        self.create_toolbar(main_container)
        
        # 메인 작업 영역
        work_area = tk.Frame(main_container)
        work_area.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # 좌측: 이미지 캔버스
        self.create_canvas_area(work_area)
        
        # 우측: 컨트롤 패널
        self.create_control_panel(work_area)
        
    def create_toolbar(self, parent):
        """상단 툴바 생성"""
        toolbar = tk.Frame(parent, bg='lightgray', height=50)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        toolbar.pack_propagate(False)
        
        # 파일 관련 버튼들
        tk.Button(toolbar, text="📁 폴더 선택", command=self.select_folder, 
                 bg='lightblue', font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=5, pady=10)
        
        tk.Button(toolbar, text="🖼️ 이미지 로드", command=self.load_image, 
                 bg='lightgreen', font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=5, pady=10)
        
        # 진행률 표시
        self.progress_label = tk.Label(toolbar, text="진행률: 0/0", 
                                     font=('Arial', 11), bg='lightgray')
        self.progress_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # 네비게이션 버튼
        tk.Button(toolbar, text="⬅️ 이전", command=self.prev_image, 
                 font=('Arial', 10)).pack(side=tk.RIGHT, padx=5, pady=10)
        tk.Button(toolbar, text="➡️ 다음", command=self.next_image, 
                 font=('Arial', 10)).pack(side=tk.RIGHT, padx=5, pady=10)
        
    def create_canvas_area(self, parent):
        """이미지 캔버스 영역 생성"""
        canvas_frame = tk.Frame(parent, bg='white', relief=tk.SUNKEN, bd=2)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 캔버스
        self.canvas = tk.Canvas(canvas_frame, bg='white', cursor='crosshair')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 마우스 이벤트 바인딩
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        
        # 키보드 단축키
        self.root.bind("<Key>", self.on_key_press)
        self.root.focus_set()
        
    def create_control_panel(self, parent):
        """우측 컨트롤 패널 생성"""
        control_frame = tk.Frame(parent, width=350, bg='lightgray', relief=tk.RAISED, bd=2)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        control_frame.pack_propagate(False)
        
        # 제목
        tk.Label(control_frame, text="🎯 컨트롤 패널", 
                font=('Arial', 14, 'bold'), bg='lightgray').pack(pady=10)
        
        # 클래스 선택 영역
        self.create_class_selection(control_frame)
        
        # 어노테이션 리스트 영역
        self.create_annotation_list(control_frame)
        
        # 액션 버튼들
        self.create_action_buttons(control_frame)
        
        # 도움말
        self.create_help_section(control_frame)
        
    def create_class_selection(self, parent):
        """클래스 선택 UI"""
        class_frame = tk.LabelFrame(parent, text="클래스 선택", 
                                   font=('Arial', 12, 'bold'), bg='lightgray')
        class_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.class_var = tk.StringVar(value=CLASSES[0])
        
        for i, class_name in enumerate(CLASSES):
            color = COLORS[i % len(COLORS)]
            
            frame = tk.Frame(class_frame, bg='lightgray')
            frame.pack(fill=tk.X, pady=2)
            
            btn = tk.Radiobutton(frame, text=f"{i+1}. {class_name}", 
                               variable=self.class_var, value=class_name,
                               command=lambda idx=i: setattr(self, 'current_class', idx),
                               font=('Arial', 11), fg=color, bg='lightgray',
                               selectcolor='white')
            btn.pack(side=tk.LEFT)
            
            # 단축키 표시
            tk.Label(frame, text=f"[{i+1}]", font=('Arial', 9), 
                    fg='gray', bg='lightgray').pack(side=tk.RIGHT)
    
    def create_annotation_list(self, parent):
        """어노테이션 리스트 UI"""
        list_frame = tk.LabelFrame(parent, text="현재 어노테이션", 
                                  font=('Arial', 12, 'bold'), bg='lightgray')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 리스트박스와 스크롤바
        list_container = tk.Frame(list_frame, bg='lightgray')
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.annotation_listbox = tk.Listbox(list_container, 
                                           yscrollcommand=scrollbar.set,
                                           font=('Arial', 10),
                                           selectmode=tk.SINGLE)
        self.annotation_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.annotation_listbox.yview)
        
        # 리스트 관리 버튼
        list_btn_frame = tk.Frame(list_frame, bg='lightgray')
        list_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(list_btn_frame, text="삭제", command=self.delete_annotation,
                 bg='lightcoral', font=('Arial', 10)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        tk.Button(list_btn_frame, text="모두 삭제", command=self.clear_annotations,
                 bg='red', fg='white', font=('Arial', 10)).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
    
    def create_action_buttons(self, parent):
        """액션 버튼들"""
        action_frame = tk.LabelFrame(parent, text="액션", 
                                   font=('Arial', 12, 'bold'), bg='lightgray')
        action_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(action_frame, text="💾 저장", command=self.save_annotations,
                 bg='darkgreen', fg='white', font=('Arial', 12, 'bold')).pack(fill=tk.X, pady=5)
        
        tk.Button(action_frame, text="↻ 새로고침", command=self.refresh_display,
                 bg='orange', font=('Arial', 10)).pack(fill=tk.X, pady=2)
        
        tk.Button(action_frame, text="📊 통계", command=self.show_statistics,
                 bg='purple', fg='white', font=('Arial', 10)).pack(fill=tk.X, pady=2)
    
    def create_help_section(self, parent):
        """도움말 섹션"""
        help_frame = tk.LabelFrame(parent, text="사용법", 
                                  font=('Arial', 12, 'bold'), bg='lightgray')
        help_frame.pack(fill=tk.X, padx=10, pady=5)
        
        help_text = """
🖱️ 마우스 드래그: 바운딩 박스 그리기
⌨️ 숫자키 1-5: 클래스 선택
⌨️ S: 저장
⌨️ ←→: 이미지 이동
⌨️ Delete: 선택된 박스 삭제
⌨️ Ctrl+A: 모든 박스 삭제
        """
        
        tk.Label(help_frame, text=help_text.strip(), justify=tk.LEFT, 
                font=('Arial', 9), bg='lightyellow', fg='black').pack(fill=tk.X, padx=5, pady=5)
    
    def on_key_press(self, event):
        """키보드 단축키 처리"""
        key = event.keysym
        
        # 숫자키로 클래스 선택
        if key.isdigit():
            class_idx = int(key) - 1
            if 0 <= class_idx < len(CLASSES):
                self.current_class = class_idx
                self.class_var.set(CLASSES[class_idx])
        
        # 기능키
        elif key == 's' or key == 'S':
            self.save_annotations()
        elif key == 'Left':
            self.prev_image()
        elif key == 'Right':
            self.next_image()
        elif key == 'Delete':
            self.delete_annotation()
        elif event.state & 0x4 and key == 'a':  # Ctrl+A
            self.clear_annotations()
    
    def select_folder(self):
        """이미지 폴더 선택"""
        folder = filedialog.askdirectory(
            title="이미지 폴더 선택",
            initialdir=str(PATHS['images_dir'])
        )
        
        if folder:
            self.image_folder = Path(folder)
            
            # 이미지 파일 목록 생성
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            self.image_files = []
            
            for ext in extensions:
                self.image_files.extend(self.image_folder.glob(f'*{ext}'))
                self.image_files.extend(self.image_folder.glob(f'*{ext.upper()}'))
            
            self.image_files.sort()
            self.current_image_index = 0
            
            messagebox.showinfo("성공", f"{len(self.image_files)}개의 이미지를 찾았습니다.")
            self.update_progress()
            
            if self.image_files:
                self.load_image()
    
    def load_image(self):
        """현재 인덱스의 이미지 로드"""
        if not hasattr(self, 'image_files') or not self.image_files:
            # 단일 이미지 선택
            image_path = filedialog.askopenfilename(
                title="이미지 선택",
                initialdir=str(PATHS['images_dir']),
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
            )
            if image_path:
                self.load_specific_image(image_path)
        else:
            if self.current_image_index < len(self.image_files):
                image_path = self.image_files[self.current_image_index]
                self.load_specific_image(str(image_path))
                self.update_progress()
            else:
                messagebox.showinfo("완료", "모든 이미지 라벨링이 완료되었습니다!")
    
    def load_specific_image(self, image_path):
        """특정 이미지 로드"""
        try:
            self.current_image_path = image_path
            
            # 이미지 로드
            image = Image.open(image_path)
            
            # 캔버스 크기에 맞게 조정
            canvas_width = self.canvas.winfo_width() or 900
            canvas_height = self.canvas.winfo_height() or 600
            
            # 비율 유지하면서 크기 조정
            image.thumbnail((canvas_width-20, canvas_height-20), Image.Resampling.LANCZOS)
            
            self.display_image = ImageTk.PhotoImage(image)
            self.original_image = image
            
            # 스케일 팩터 계산
            orig_img = Image.open(image_path)
            self.scale_x = orig_img.width / image.width
            self.scale_y = orig_img.height / image.height
            
            # 캔버스에 이미지 표시
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                   anchor=tk.CENTER, image=self.display_image)
            
            # 이미지 위치 계산 (중앙 정렬)
            self.image_x = (canvas_width - image.width) // 2
            self.image_y = (canvas_height - image.height) // 2
            
            # 기존 어노테이션 로드
            self.load_existing_annotations()
            
            # 제목 업데이트
            filename = Path(image_path).name
            self.root.title(f"YOLO 라벨링 도구 - {filename}")
            
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 실패: {str(e)}")
    
    def load_existing_annotations(self):
        """기존 어노테이션 파일 로드"""
        if not self.current_image_path:
            return
            
        # annotations 폴더에서 JSON 파일 찾기
        from config import PATHS
        annotations_dir = PATHS['annotations_dir']
        image_name = Path(self.current_image_path).stem
        annotation_path = annotations_dir / f"{image_name}.json"
        
        self.annotations = []
        if annotation_path.exists():
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.annotations = data.get('annotations', [])
                self.update_annotation_display()
            except Exception as e:
                print(f"어노테이션 로드 오류: {e}")
    
    def start_draw(self, event):
        """바운딩 박스 그리기 시작"""
        # 이미지 영역 내에서만 그리기 허용
        if hasattr(self, 'image_x') and hasattr(self, 'image_y'):
            if (self.image_x <= event.x <= self.image_x + self.original_image.width and
                self.image_y <= event.y <= self.image_y + self.original_image.height):
                
                self.drawing = True
                self.start_x = event.x
                self.start_y = event.y
    
    def draw_rect(self, event):
        """바운딩 박스 그리기 중"""
        if self.drawing and self.start_x and self.start_y:
            if self.temp_rect:
                self.canvas.delete(self.temp_rect)
            
            color = COLORS[self.current_class % len(COLORS)]
            
            self.temp_rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline=color, width=3, tags="temp"
            )
    
    def end_draw(self, event):
        """바운딩 박스 그리기 종료"""
        if not self.drawing or not self.start_x or not self.start_y:
            return
            
        self.drawing = False
        
        # 최소 크기 체크
        if abs(event.x - self.start_x) < 10 or abs(event.y - self.start_y) < 10:
            if self.temp_rect:
                self.canvas.delete(self.temp_rect)
            return
        
        # 이미지 영역 확인
        if not (hasattr(self, 'image_x') and hasattr(self, 'image_y')):
            return
        
        # 이미지 상대 좌표로 변환
        x1 = max(0, min(self.start_x, event.x) - self.image_x)
        y1 = max(0, min(self.start_y, event.y) - self.image_y)
        x2 = min(self.original_image.width, max(self.start_x, event.x) - self.image_x)
        y2 = min(self.original_image.height, max(self.start_y, event.y) - self.image_y)
        
        # 원본 이미지 좌표로 변환
        orig_x1 = x1 * self.scale_x
        orig_y1 = y1 * self.scale_y
        orig_x2 = x2 * self.scale_x
        orig_y2 = y2 * self.scale_y
        
        # 어노테이션 추가
        annotation = {
            'class_id': self.current_class,
            'class_name': CLASSES[self.current_class],
            'bbox': [orig_x1, orig_y1, orig_x2, orig_y2]
        }
        
        self.annotations.append(annotation)
        self.update_annotation_display()
        
        self.temp_rect = None
    
    def update_annotation_display(self):
        """어노테이션 화면 표시 업데이트"""
        # 기존 어노테이션 박스 삭제
        for item in self.canvas.find_withtag("annotation"):
            self.canvas.delete(item)
        
        # 리스트박스 업데이트
        self.annotation_listbox.delete(0, tk.END)
        
        if not hasattr(self, 'image_x') or not hasattr(self, 'image_y'):
            return
        
        for i, ann in enumerate(self.annotations):
            # 화면 좌표로 변환
            x1 = ann['bbox'][0] / self.scale_x + self.image_x
            y1 = ann['bbox'][1] / self.scale_y + self.image_y
            x2 = ann['bbox'][2] / self.scale_x + self.image_x
            y2 = ann['bbox'][3] / self.scale_y + self.image_y
            
            color = COLORS[ann['class_id'] % len(COLORS)]
            
            # 바운딩 박스 그리기
            self.canvas.create_rectangle(x1, y1, x2, y2, 
                                       outline=color, width=3, 
                                       tags="annotation")
            
            # 클래스 라벨
            self.canvas.create_text(x1, y1-15, text=ann['class_name'], 
                                  fill=color, font=('Arial', 12, 'bold'), 
                                  anchor=tk.W, tags="annotation")
            
            # 리스트박스에 추가
            w = int(ann['bbox'][2] - ann['bbox'][0])
            h = int(ann['bbox'][3] - ann['bbox'][1])
            bbox_info = f"{i+1}. {ann['class_name']} ({w}×{h})"
            self.annotation_listbox.insert(tk.END, bbox_info)
    
    def delete_annotation(self):
        """선택된 어노테이션 삭제"""
        selection = self.annotation_listbox.curselection()
        if selection:
            index = selection[0]
            del self.annotations[index]
            self.update_annotation_display()
    
    def clear_annotations(self):
        """모든 어노테이션 삭제"""
        if self.annotations and messagebox.askyesno("확인", "모든 어노테이션을 삭제하시겠습니까?"):
            self.annotations = []
            self.update_annotation_display()
    
    def save_annotations(self):
        """어노테이션 저장"""
        if not self.current_image_path:
            messagebox.showwarning("경고", "저장할 이미지가 없습니다.")
            return
        
        # annotations 폴더에 JSON 저장
        from config import PATHS
        annotations_dir = PATHS['annotations_dir']
        annotations_dir.mkdir(exist_ok=True)
        
        image_name = Path(self.current_image_path).stem
        annotation_path = annotations_dir / f"{image_name}.json"
        
        # 원본 이미지 크기 정보
        orig_img = Image.open(self.current_image_path)
        
        data = {
            'image_path': self.current_image_path,
            'image_size': [orig_img.width, orig_img.height],
            'annotations': self.annotations,
            'total_objects': len(self.annotations)
        }
        
        try:
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("성공", f"어노테이션이 저장되었습니다!\n{annotation_path}")
            
            # 자동으로 다음 이미지로 이동 (폴더 모드일 때)
            if hasattr(self, 'image_files') and self.image_files:
                self.next_image()
                
        except Exception as e:
            messagebox.showerror("오류", f"저장 실패: {str(e)}")
    
    def prev_image(self):
        """이전 이미지"""
        if hasattr(self, 'image_files') and self.image_files:
            if self.current_image_index > 0:
                self.current_image_index -= 1
                self.load_image()
    
    def next_image(self):
        """다음 이미지"""
        if hasattr(self, 'image_files') and self.image_files:
            if self.current_image_index < len(self.image_files) - 1:
                self.current_image_index += 1
                self.load_image()
    
    def update_progress(self):
        """진행률 업데이트"""
        if hasattr(self, 'image_files') and self.image_files:
            total = len(self.image_files)
            current = self.current_image_index + 1
            self.progress_label.config(text=f"진행률: {current}/{total}")
    
    def refresh_display(self):
        """화면 새로고침"""
        if self.current_image_path:
            self.load_specific_image(self.current_image_path)
    
    def show_statistics(self):
        """라벨링 통계 표시"""
        if not self.annotations:
            messagebox.showinfo("통계", "현재 이미지에 어노테이션이 없습니다.")
            return
        
        # 클래스별 카운트
        class_counts = {}
        for ann in self.annotations:
            class_name = ann['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # 통계 메시지 생성
        stats_msg = f"현재 이미지 통계:\n\n"
        stats_msg += f"총 객체 수: {len(self.annotations)}\n\n"
        stats_msg += "클래스별 분포:\n"
        
        for class_name, count in class_counts.items():
            stats_msg += f"  - {class_name}: {count}개\n"
        
        messagebox.showinfo("통계", stats_msg)
    
    def run(self):
        """GUI 실행"""
        print("🏷️ 이미지 라벨링 도구 시작")
        print("📁 기본 이미지 폴더:", PATHS['images_dir'])
        self.root.mainloop()

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🏷️ YOLO 이미지 라벨링 도구")
    print("=" * 60)
    print(f"클래스: {', '.join(CLASSES)}")
    print(f"색상: {', '.join(COLORS[:len(CLASSES)])}")
    
    # 디렉토리 설정
    setup_directories()
    
    # 라벨링 도구 실행
    tool = ImageLabelingTool()
    tool.run()

if __name__ == "__main__":
    main()