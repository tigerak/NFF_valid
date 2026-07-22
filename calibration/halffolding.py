import sys, cv2, os, json, yaml, time
import numpy as np
from datetime import datetime

qt_bin_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'bin')
if os.path.exists(qt_bin_path):
    os.add_dll_directory(qt_bin_path)

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from scipy.optimize import minimize

# --- YAML 리스트 형식을 강제하기 위한 커스텀 클래스 ---
class SequenceWrapper(list):
    pass
def sequence_representer(dumper, data):
    # 이 클래스로 감싸진 리스트만 가로([ ]) 스타일로 출력하게 설정
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
yaml.add_representer(SequenceWrapper, sequence_representer)

# --- 1. 최적화 과정 시각화 전용 창 (동일) ---
class MonitorWindow(QMainWindow):
    def __init__(self, base_pixmap, e_pts, s_pts):
        super().__init__()
        self.setWindowTitle("3D Optimization Real-time Monitor")
        self.base_pixmap = base_pixmap.copy() if base_pixmap else None
        
        # [추가] 초기 가이드라인 그리기 (하얀색)
        if self.base_pixmap:
            painter = QPainter(self.base_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # 1. 초기 타원 피팅 가이드 (하얀색 실선)
            if len(e_pts) >= 5:
                try:
                    el = cv2.fitEllipse(e_pts.astype(np.float32))
                    painter.setPen(QPen(QColor(255, 255, 255, 150), 2)) # 반투명 하얀색
                    painter.save()
                    painter.translate(el[0][0], el[0][1])
                    painter.rotate(el[2])
                    painter.drawEllipse(QRectF(-el[1][0]/2, -el[1][1]/2, el[1][0], el[1][1]))
                    painter.restore()
                except: pass
            
            # 2. 사용자가 지정한 수직축 가이드 (하얀색 점선)
            if len(s_pts) >= 2:
                painter.setPen(QPen(QColor(255, 255, 255, 150), 1, Qt.DashLine))
                painter.drawLine(QPointF(*s_pts[0]), QPointF(*s_pts[-1]))
            
            painter.end()

        self.display_label = QLabel("Starting Optimization...")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.display_label)
        self.resize(1000, 800)

    def update_view(self, top_pts, axis_pts, e_pts, s_pts, loss):
        if self.base_pixmap is None: return
        canvas = self.base_pixmap.copy()
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 사용자 클릭 데이터 표시
        painter.setPen(QPen(Qt.red, 10)); [painter.drawPoint(QPointF(*p)) for p in e_pts]
        painter.setPen(QPen(Qt.cyan, 10)); [painter.drawPoint(QPointF(*p)) for p in s_pts]
        
        # [추가] 사용자가 찍은 옆면 4점 연결 (하얀 점선 가이드)
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1, Qt.DashLine))
        if len(s_pts) >= 2: # 첫 번째 라인 (좌측)
            painter.drawLine(QPointF(*s_pts[0]), QPointF(*s_pts[1]))
        if len(s_pts) >= 4: # 두 번째 라인 (우측)
            painter.drawLine(QPointF(*s_pts[2]), QPointF(*s_pts[3]))

        # 2. [모델] 상단 타원 (Yellow)
        if len(top_pts) > 0:
            painter.setPen(QPen(QColor(255, 255, 0), 3))
            painter.drawPolyline(QPolygonF([QPointF(*p) for p in top_pts]))
        
        # 3. [모델] 중심축 (Green)
        if len(axis_pts) >= 2:
            painter.setPen(QPen(Qt.green, 4))
            painter.drawLine(QPointF(*axis_pts[0]), QPointF(*axis_pts[1]))

        # 4. [모델] 옆면 외곽선 투영 (Orange)
        # 3D 모델의 실제 외곽선 2개를 계산하여 화면에 그려주면 정밀도가 보입니다.
        # (이 부분은 objective 함수에서 외곽선 좌표를 넘겨주어야 정확합니다.)

        painter.setPen(QPen(Qt.white, 2))
        painter.drawText(30, 50, f"Total Loss: {loss:.6f}")
        painter.end()
        self.display_label.setPixmap(canvas.scaled(self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        QApplication.processEvents()

# --- 2. 메인 GUI 클래스 ---
class CalibratorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.settings_file = "calib_settings.json"
        self.data_file = "collected_points.json"
        
        self.image_list = []; self.current_idx = -1
        self.original_cv_img = None; self.display_pixmap = None
        self.points_ellipse = []; self.points_side = []
        self.current_mode = "ellipse"
        
        # 설정 로드
        self.settings = self.load_settings()
        self.focal_length = self.settings.get("focal_length", 3000)
        self.real_diameter = self.settings.get("real_diameter", 100.0)
        self.initial_distance = self.settings.get("initial_distance", 1500.0) # TZ 초기값 로드
        self.enhance_val = self.settings.get("enhance_value", 10)
        self.axis_length = self.settings.get("axis_length", 25)

        self.cyl_height = self.settings.get("cyl_height", 20.0)  # <-- 이 줄을 추가하세요!
        self.cyl_diameter = self.settings.get("cyl_diameter", 50)

        self.initUI()
        last_path = self.settings.get("last_path", "")
        if last_path and os.path.exists(last_path): self.open_file(last_path)

    def initUI(self):
        self.setStyleSheet("QWidget { background-color: #222; color: #EEE; font-family: 'Malgun Gothic'; }")
        main_layout = QHBoxLayout()

        # [좌측: 이미지]
        left_layout = QVBoxLayout()
        self.path_display = QLineEdit(); self.path_display.setReadOnly(True)
        self.img_display = QLabel("No Image Loaded"); self.img_display.setFixedSize(900, 700)
        self.img_display.setStyleSheet("background-color: #000; border: 1px solid #444;")
        self.img_display.setAlignment(Qt.AlignCenter)
        self.img_display.mousePressEvent = self.on_mouse_click

        # --- 이미지 탐색 컨트롤 영역 ---
        nav_layout = QHBoxLayout()
        
        # 버튼 스타일 정의 (가독성을 위해 크게 설정)
        btn_style = "font-weight: bold; min-width: 40px; height: 30px;"
        
        # 처음으로 이동
        self.btn_first = QPushButton("|<")
        self.btn_first.clicked.connect(lambda: self.jump_image('first'))
        
        # 100장 뒤로
        self.btn_prev_100 = QPushButton("<<<")
        self.btn_prev_100.clicked.connect(lambda: self.jump_image(-100))
        
        # 10장 뒤로
        self.btn_prev_10 = QPushButton("<<")
        self.btn_prev_10.clicked.connect(lambda: self.jump_image(-10))
        
        # 1장 뒤로
        self.btn_prev_1 = QPushButton("<")
        self.btn_prev_1.clicked.connect(lambda: self.jump_image(-1))

        # 현재 프레임 표시 (중앙)
        self.lbl_frame_info = QLabel("0 / 0")
        self.lbl_frame_info.setAlignment(Qt.AlignCenter)
        self.lbl_frame_info.setStyleSheet("font-size: 14px; font-weight: bold; min-width: 120px; color: #EEE;")

        # 1장 앞으로
        self.btn_next_1 = QPushButton(">")
        self.btn_next_1.clicked.connect(lambda: self.jump_image(1))
        
        # 10장 앞으로
        self.btn_next_10 = QPushButton(">>")
        self.btn_next_10.clicked.connect(lambda: self.jump_image(10))
        
        # 100장 앞으로
        self.btn_next_100 = QPushButton(">>>")
        self.btn_next_100.clicked.connect(lambda: self.jump_image(100))
        
        # 끝으로 이동
        self.btn_last = QPushButton(">|")
        self.btn_last.clicked.connect(lambda: self.jump_image('last'))

        # 버튼들을 레이아웃에 추가
        nav_widgets = [
            self.btn_first, self.btn_prev_100, self.btn_prev_10, self.btn_prev_1,
            self.lbl_frame_info,
            self.btn_next_1, self.btn_next_10, self.btn_next_100, self.btn_last
        ]
        for w in nav_widgets:
            if isinstance(w, QPushButton): w.setStyleSheet(btn_style)
            nav_layout.addWidget(w)

        left_layout.addWidget(self.path_display); left_layout.addWidget(self.img_display); left_layout.addLayout(nav_layout)
        main_layout.addLayout(left_layout, stretch=3)

        # [우측: 설정 패널]
        right_panel = QGroupBox("Cylinder Optimization"); right_panel.setFixedWidth(320)
        ctrl = QVBoxLayout()

        # 2. 기존 레이아웃(예: self.ctrl_layout)의 상단에 추가
        # 기존 코드에서 레이아웃 변수명을 찾아 'addWidget' 하시면 됩니다.
        self.btn_open = QPushButton("📂 OPEN IMAGE"); self.btn_open.clicked.connect(lambda: self.open_file())
        self.btn_open.setStyleSheet("background-color: #2E7D32; color: white; height: 40px;")
        ctrl.addWidget(self.btn_open)
        #ctrl.addSpacing(15)  # 15픽셀 정도 띄우기


        # 2. 이미지 인핸스 (Contrast & Brightness)
        # 중요: 라벨을 슬라이더보다 먼저 정의해야 합니다.
        self.enhance_label = QLabel(f"Contrast: {self.enhance_val / 10.0:.1f}x")
        ctrl.addWidget(self.enhance_label)
        self.enhance_slider = QSlider(Qt.Horizontal)
        self.enhance_slider.setRange(5, 100)  # 0.5배 ~ 10.0배
        self.enhance_slider.setValue(self.enhance_val)
        self.enhance_slider.valueChanged.connect(self.apply_enhance)
        ctrl.addWidget(self.enhance_slider)
        self.brightness_label = QLabel("Brightness: 0")
        ctrl.addWidget(self.brightness_label)
        self.bright_slider = QSlider(Qt.Horizontal)
        self.bright_slider.setRange(-100, 100)
        self.bright_slider.setValue(0)
        self.bright_slider.valueChanged.connect(self.apply_enhance)
        ctrl.addWidget(self.bright_slider)
        ctrl.addSpacing(10)


        self.rb_ellipse = QRadioButton("타원 점 (Top)"); self.rb_side = QRadioButton("옆면 점 (Side)")
        self.rb_ellipse.setChecked(True); self.rb_ellipse.toggled.connect(self.change_mode)
        ctrl.addWidget(self.rb_ellipse); ctrl.addWidget(self.rb_side)
        btn_clear = QPushButton("🗑 좌표 초기화"); btn_clear.clicked.connect(self.clear_pts)
        ctrl.addWidget(btn_clear)
        self.btn_save = QPushButton("💾 SAVE DATA"); self.btn_save.clicked.connect(self.save_data)
        ctrl.addWidget(self.btn_save)

        # 1. 수치 입력창들
        self.btn_run = QPushButton("🚀 RUN CALIBRATION"); self.btn_run.clicked.connect(self.run_calibration)
        self.btn_run.setStyleSheet("background-color: #2E7D32; height: 40px; font-weight: bold;")
        ctrl.addWidget(self.btn_run)
        ctrl.addWidget(QLabel("Focal Length (px):"))
        self.focal_input = QLineEdit(str(self.focal_length)); self.focal_input.textChanged.connect(self.save_settings)
        ctrl.addWidget(self.focal_input)
        ctrl.addWidget(QLabel("Real Diameter (mm):"))
        self.diameter_input = QLineEdit(str(self.real_diameter)); self.diameter_input.textChanged.connect(self.save_settings)
        ctrl.addWidget(self.diameter_input)
        ctrl.addWidget(QLabel("Initial Distance (TZ, mm):"))
        self.initial_dist_input = QLineEdit(str(self.initial_distance)); self.initial_dist_input.textChanged.connect(self.save_settings)
        ctrl.addWidget(self.initial_dist_input)
        ctrl.addSpacing(10)


        # 오버레이 제어 변수
        self.last_calib_params = None  # [rx, ry, rz, tx, ty, tz] 저장
        self.is_overlay_on = False     # 오버레이 활성화 상태
        
        # 오버레이 토글 버튼 추가
        self.btn_overlay = QPushButton("👁️ 결과 오버레이 ON/OFF")
        self.btn_overlay.setCheckable(True)
        self.btn_overlay.clicked.connect(self.toggle_overlay)
        self.btn_overlay.setStyleSheet("background-color: #555; color: white; height: 40px;")
        ctrl.addWidget(self.btn_overlay)
        


        ctrl.addWidget(QLabel("Axis Length (mm) for display:"))
        self.axis_len_input = QLineEdit(str(self.axis_length)); self.axis_len_input.textChanged.connect(self.save_settings)
        ctrl.addWidget(self.axis_len_input)

        # --- [Virtual Cylinder Diameter 조절 세트] ---
        ctrl.addWidget(QLabel("Virtual Cylinder Diameter (mm):")) # 레이블 추가
        self.dia_slider = QSlider(Qt.Horizontal)
        self.dia_slider.setRange(10, 500) # 1.0mm ~ 500.0mm (10배 곱해서 정수로 관리)
        self.dia_slider.setValue(int(self.cyl_diameter * 10))
        self.dia_slider.valueChanged.connect(self.sync_dia_slider_to_spin)
        self.dia_spin = QDoubleSpinBox() # 텍스트 입력 및 위/아래 버튼 역할
        self.dia_spin.setRange(1.0, 50.0)
        self.dia_spin.setSingleStep(0.1)
        self.dia_spin.setValue(self.cyl_diameter)
        self.dia_spin.setFixedWidth(80)
        self.dia_spin.valueChanged.connect(self.sync_dia_spin_to_slider)
        dia_container = QHBoxLayout() # 슬라이더와 스핀박스를 가로로 묶음
        dia_container.addWidget(self.dia_slider)
        dia_container.addWidget(self.dia_spin)
        ctrl.addLayout(dia_container) # 메인 레이아웃에 가로 세트 추가

        # --- [Virtual Cylinder Height 조절 세트] ---
        ctrl.addWidget(QLabel("Virtual Cylinder Height (mm):")) # 레이블 추가
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setRange(10, 1000) # 1.0mm ~ 1000.0mm
        self.height_slider.setValue(int(self.cyl_height * 10))
        self.height_slider.valueChanged.connect(self.sync_height_slider_to_spin)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1.0, 100.0)
        self.height_spin.setSingleStep(1.0)
        self.height_spin.setValue(self.cyl_height)
        self.height_spin.setFixedWidth(80)
        self.height_spin.valueChanged.connect(self.sync_height_spin_to_slider)
        height_layout = QHBoxLayout()
        height_layout.addWidget(self.height_slider)
        height_layout.addWidget(self.height_spin)
        height_container = QHBoxLayout() # 슬라이더와 스핀박스를 가로로 묶음
        height_container.addWidget(self.height_slider)
        height_container.addWidget(self.height_spin)
        ctrl.addLayout(height_container) # 메인 레이아웃에 가로 세트 추가
        ctrl.addSpacing(10)


        # [버튼 생성 - initUI 내부에 추가]
        self.btn_save_yaml = QPushButton("💾 YAML 결과 저장")
        self.btn_save_yaml.clicked.connect(self.save_calibration_yaml)
        self.btn_save_yaml.setStyleSheet("background-color: #1976D2; height: 40px; font-weight: bold;")
        ctrl.addWidget(self.btn_save_yaml)

        # 버튼 생성
        self.btn_load_yaml = QPushButton("Load YAML Result")
        self.btn_load_yaml.clicked.connect(self.load_calibration_yaml)
        self.btn_load_yaml.setStyleSheet("background-color: #1976D2; height: 40px; font-weight: bold;")

        # 레이아웃에 추가 (저장 버튼 근처에 배치)
        ctrl.addWidget(self.btn_load_yaml)

        ctrl.addStretch(); right_panel.setLayout(ctrl); main_layout.addWidget(right_panel)
        self.setLayout(main_layout)
        self.setWindowTitle("Cylinder 6-DOF System v9.6")
        
        # 창을 화면 중앙으로 이동
        self.move_to_top()


    # --- Diameter 동기화 ---
    def sync_dia_slider_to_spin(self, val):
        new_val = val / 10.0
        if self.dia_spin.value() != new_val:
            self.dia_spin.setValue(new_val)
        self.save_settings()
        self.update_display()

    def sync_dia_spin_to_slider(self, val):
        new_slider_val = int(val * 10)
        if self.dia_slider.value() != new_slider_val:
            self.dia_slider.setValue(new_slider_val)
        self.save_settings()
        self.update_display()

    def load_calibration_yaml(self):
        """저장된 YAML 파일을 읽어 화면에 복원"""
        # 1. 파일 선택 창 열기
        file_path, _ = QFileDialog.getOpenFileName(
            self, "캘리브레이션 결과 불러오기", "result", "YAML Files (*.yaml)"
        )
        
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if not data:
                return

            # --- [추가] 이미지 일치 확인 및 로드 로직 ---
            yaml_image_path = data.get("camera_info", {}).get("image_path", "")
            
            # 현재 열린 이미지와 YAML의 이미지가 다를 경우
            current_img = self.image_list[self.current_idx] if self.image_list else ""
            
            if yaml_image_path and os.path.normpath(yaml_image_path) != os.path.normpath(current_img):
                reply = QMessageBox.question(self, "이미지 변경 확인", 
                                        "YAML에 기록된 이미지와 현재 이미지가 다릅니다.\n"
                                        "해당 이미지를 새로 불러올까요?",
                                        QMessageBox.Yes | QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    if os.path.exists(yaml_image_path):
                        # 이미지를 새로 열면 앞서 만든 open_file(path)이 실행되어 
                        # 리스트 구축 및 초기화가 진행됩니다.
                        self.open_file(yaml_image_path)
                    else:
                        QMessageBox.warning(self, "파일 없음", "YAML에 기록된 이미지 경로를 찾을 수 없습니다.")
                        return # 이미지 없으면 데이터 로드 중단
                else:
                    return # 사용자가 취소하면 중단
            
            # 2. 물리 설정(Diameter, Height 등) 복원
            # YAML에는 Meter 단위로 저장했으므로 다시 mm로 변환 (* 1000)
            p_set = data.get("physical_settings", {})
            
            if "real_diameter" in p_set:
                val = p_set["real_diameter"] * 1000.0
                self.dia_spin.setValue(val) # 스핀박스 수정 (연결된 슬라이더도 자동 변경됨)
                
            if "initial_distance" in p_set: # YAML에 height를 이 이름으로 저장했다면
                val = p_set["initial_distance"] * 1000.0
                self.height_spin.setValue(val)

            # 3. 최적화 결과(Extrinsic) 복원
            if "CameraParameterForHalfFolding" in data:
                # YAML의 리스트를 다시 numpy 배열로 변환하여 보관
                self.last_calib_params = np.array(data["CameraParameterForHalfFolding"]["extrinsic_parameter"])
                # Extrinsic 값은 단위 변환(m->mm, deg->rad)이 필요할 수 있으니 주의하세요.
                self.last_calib_params[:3] = np.deg2rad(self.last_calib_params[:3])
                self.last_calib_params[3:] *= 1000.

            # 4. (선택사항) 포인트 정보가 YAML에 있다면 복원
            # 만약 저장할 때 포인트 좌표도 넣으셨다면 여기서 self.points_ellipse 등에 할당
            pts = data.get("points", {})
            # [ [x,y], [x,y] ] 형태를 다시 리스트로 저장
            self.points_ellipse = pts.get("ellipse", [])
            self.points_side = pts.get("side", [])

            # 5. 화면 새로고침 (이 함수가 paintEvent를 트리거하여 점을 다시 그립니다)
            # 1. 오버레이 체크박스가 있다면 강제로 체크
            if hasattr(self, 'btn_overlay'):
                self.btn_overlay.setChecked(True)
            
            # 2. 내부 오버레이 플래그 변수가 있다면 True로 설정
            self.is_overlay_on = True
            self.update_display()
            QMessageBox.information(self, "불러오기 완료", f"{os.path.basename(file_path)} 데이터를 불러왔습니다.")

        except Exception as e:
            QMessageBox.critical(self, "불러오기 실패", f"파일을 읽는 중 오류가 발생했습니다: {str(e)}")

            
    # --- Height 동기화 ---
    def sync_height_slider_to_spin(self, val):
        new_val = val / 10.0
        if self.height_spin.value() != new_val:
            self.height_spin.setValue(new_val)
        self.save_settings()
        self.update_display()

    def sync_height_spin_to_slider(self, val):
        new_slider_val = int(val * 10)
        if self.height_slider.value() != new_slider_val:
            self.height_slider.setValue(new_slider_val)
        self.save_settings()
        self.update_display()


    def jump_image(self, step):
        if not self.image_list: return
        
        total = len(self.image_list)
        
        if step == 'first':
            self.current_idx = 0
        elif step == 'last':
            self.current_idx = total - 1
        else:
            # -100, -10, -1, 1, 10, 100 단위 이동
            self.current_idx += step
            
        # 범위 초과 방지 (Clip)
        if self.current_idx < 0: self.current_idx = 0
        if self.current_idx >= total: self.current_idx = total - 1
        
        self.load_image() # 이미지를 실제로 불러오는 함수 호출
        self.update_frame_label() # 라벨 갱신

    def update_frame_label(self):
        """중앙 프레임 정보를 갱신 (예: [ 105 / 2000 ])"""
        if self.image_list:
            info = f"[ {self.current_idx + 1} / {len(self.image_list)} ]"
            self.lbl_frame_info.setText(info)

    def move_to_top(self):
        """창을 모니터 화면의 최상단 중앙에 위치시킴"""
        # 현재 화면의 사용 가능한 영역(작업표시줄 제외) 정보 가져오기
        # PyQt5 기준 (PyQt6라면 self.screen().availableGeometry())
        screen_geo = QDesktopWidget().availableGeometry()
        
        # 창의 현재 너비와 높이 가져오기
        window_width = self.frameGeometry().width()
        
        # X 좌표: (화면 너비 - 창 너비) / 2 -> 중앙
        # Y 좌표: 0 (또는 살짝 여유를 두고 10) -> 최상단
        x = (screen_geo.width() - window_width) // 2
        y = screen_geo.top() # 화면의 가장 윗부분
        
        self.move(x, y)

    def toggle_overlay(self):
        if self.last_calib_params is None:
            QMessageBox.warning(self, "데이터 없음", "먼저 최적화를 한 번 실행해야 합니다.")
            self.btn_overlay.setChecked(False)
            return
        
        self.is_overlay_on = self.btn_overlay.isChecked()
        if self.is_overlay_on:
            self.btn_overlay.setStyleSheet("background-color: #0078D7; color: white; height: 40px;")
        else:
            self.btn_overlay.setStyleSheet("background-color: #555; color: white; height: 40px;")
        
        self.update_display() # 화면 갱신하여 그리기/지우기

    # --- 기능 함수 ---
    def load_settings(self):
        if os.path.exists(self.settings_file):
            try: return json.load(open(self.settings_file, 'r'))
            except: pass
        return {"focal_length": 3000, "real_diameter": 100.0, "initial_distance": 1500.0, "enhance_value": 10, "axis_length":150}

    def save_settings(self):
        try:
            # 현재 열려 있는 이미지의 전체 경로를 저장
            current_img_path = self.image_list[self.current_idx] if self.current_idx >= 0 else ""
            
            data = {
                "focal_length": float(self.focal_input.text() or 3000),
                "real_diameter": float(self.diameter_input.text() or 100),
                "initial_distance": float(self.initial_dist_input.text() or 1500),
                "enhance_value": self.enhance_slider.value(),
                "last_path": current_img_path,  # 최근 경로 저장
                "window_x": self.pos().x(),
                "window_y": self.pos().y(),
                "cyl_height": self.height_spin.value(),
                "cyl_diameter": self.dia_spin.value(),
            }
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Settings Save Error: {e}")


    def save_calibration_yaml(self):
        if not hasattr(self, 'last_calib_params') or self.last_calib_params is None:
            return QMessageBox.warning(self, "저장 불가", "최적화 결과가 없습니다.")

        # 1. 현재 이미지 파일명 추출 (예: 'image_01.png' -> 'image_01')
        if not self.image_list or self.current_idx < 0:
            return QMessageBox.warning(self, "오류", "열려 있는 이미지가 없습니다.")
        
        full_path = self.image_list[self.current_idx]
        # 2. 경로 구분자 및 특수문자 변환
        # :, /, \ 를 모두 _ 로 치환합니다.
        safe_name = full_path.replace(":", "").replace("/", "_").replace("\\", "_")
        # 확장자 제거 (선택 사항: .png 같은 확장자도 이름에 포함하려면 이 줄은 빼세요)
        safe_name = os.path.splitext(safe_name)[0]
    
        # 1. 경로 설정
        date_str = datetime.now().strftime("%Y-%m-%d")
        folder_path = os.path.join("result", date_str)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{safe_name}.yaml")

        # 2. 데이터 추출 및 단위 변환 (mm -> meter, rad -> degree)
        raw = self.last_calib_params 
        extrinsic_list = SequenceWrapper([
            round(float(np.degrees(raw[0])), 4),
            round(float(np.degrees(raw[1])), 4),
            round(float(np.degrees(raw[2])), 4),
            round(float(raw[3]) / 1000.0, 6),
            round(float(raw[4]) / 1000.0, 6),
            round(float(raw[5]) / 1000.0, 6)
        ])

        # 2. 전체 데이터 구성
        yaml_data = {
            "camera_info": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": self.image_list[self.current_idx] if self.image_list else ""
            },
            "points": {
                    # 리스트 내의 [x, y] 좌표들을 순수 파이썬 리스트로 변환하여 저장
                    "ellipse": [ [float(pt[0]), float(pt[1])] for pt in self.points_ellipse ],
                    "side": [ [float(pt[0]), float(pt[1])] for pt in self.points_side ]
            },
            "CameraParameterForHalfFolding": {
                "extrinsic_parameter": extrinsic_list, # SequenceWrapper가 적용됨
            },
            "physical_settings": {
                "focal_length": float(self.focal_input.text() or 0),
                "real_diameter": float(self.diameter_input.text() or 0) / 1000.0,
                "initial_distance": float(self.initial_dist_input.text() or 0) / 1000.0
            }
        }
        try:
            # 3. 저장 (default_flow_style=False로 두어 나머지는 늘여 쓰게 함)
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

            QMessageBox.information(self, "저장 완료", f"YAML 저장 성공:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", str(e))

    def apply_enhance(self):
        if self.original_cv_img is None: return
        
        # 슬라이더 값 읽기
        alpha = self.enhance_slider.value() / 10.0  # 대비 (0.5 ~ 10.0)
        beta = self.bright_slider.value()           # 밝기 (-100 ~ 100)
        
        # 라벨 업데이트
        self.enhance_label.setText(f"Contrast: {alpha:.1f}x")
        self.brightness_label.setText(f"Brightness: {beta}")
        
        # OpenCV 이미지 연산: dst = src * alpha + beta
        enhanced = cv2.convertScaleAbs(self.original_cv_img, alpha=alpha, beta=beta)
        
        # 화면 표시용 변환
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        self.display_pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, c*w, QImage.Format_RGB888).copy())
        self.update_display()
        
        # 설정 저장 (밝기는 선택사항이므로 contrast만 우선 저장)
        self.enhance_val = self.enhance_slider.value()
        self.save_settings()

    def clear_pts(self):
        self.points_ellipse = []; self.points_side = []; self.update_display()

    def clear_calibration_data(self):
        """새 이미지를 열 때 기존 데이터를 깨끗이 비움"""
        self.points_ellipse = []  # 타원(Top) 포인트 초기화
        self.points_side = []     # 옆면(Side) 포인트 초기화
        self.last_calib_params = None  # 이전 최적화 결과 삭제
        
        # UI 요소 초기화 (필요한 경우)
        if hasattr(self, 'lbl_frame_info'):
            # 최적화 결과 표시용 레이블 등이 있다면 여기서 초기화
            pass
        print("Calibration data has been reset for the new image.")

    def open_file(self, target=None):
        # 1. 시작 경로 결정: 마지막 작업 경로가 있으면 해당 폴더에서 시작, 없으면 현재 폴더
        initial_dir = ""
        if hasattr(self, 'settings') and "last_path" in self.settings:
            last_path = self.settings["last_path"]
            if os.path.exists(last_path):
                # 파일 경로가 저장되어 있다면 폴더 경로만 추출
                initial_dir = os.path.dirname(os.path.normpath(last_path))

        # 2. 파일 다이얼로그 열기 (initial_dir 적용)
        path = target if target else QFileDialog.getOpenFileName(
            self, "이미지 선택", initial_dir, "Images (*.bmp *.jpg *.png)")[0]
        
        if not path: 
            return

        # --- 핵심: 새 이미지를 로드하기 직전에 기존 데이터 초기화 ---
        self.clear_calibration_data()
        self.rb_ellipse.setChecked(True)

        # 3. 폴더 내 모든 이미지 리스트 업
        folder = os.path.dirname(os.path.normpath(path))
        self.image_list = [
            os.path.normpath(os.path.join(folder, f)) 
            for f in os.listdir(folder) 
            if f.lower().endswith(('.bmp', '.jpg', '.png'))
        ]
        self.image_list.sort()

        # 4. 현재 인덱스 설정 및 이미지 로드
        if os.path.normpath(path) in self.image_list:
            self.current_idx = self.image_list.index(os.path.normpath(path))
            self.load_image()
            # 이미지를 성공적으로 열었다면 세이브 데이터 갱신
            self.save_settings()

    def load_image(self):
        if self.current_idx < 0: return
        p = self.image_list[self.current_idx]; self.path_display.setText(p)
        try:
            arr = np.fromfile(p, np.uint8); self.original_cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            self.apply_enhance()
            self.load_saved_points(p)
            self.lbl_frame_info.setText(f"{self.current_idx + 1} / {len(self.image_list)}")
            self.save_settings()
        except Exception as e: print(f"Load Error: {e}")

    def load_saved_points(self, p):
        self.points_ellipse = []; self.points_side = []
        if os.path.exists(self.data_file):
            try:
                data = json.load(open(self.data_file, 'r', encoding='utf-8'))
                info = data.get(os.path.basename(p), {})
                self.points_ellipse = info.get("ellipse", []); self.points_side = info.get("side", [])
            except: pass

    def save_data(self):
        data = json.load(open(self.data_file, 'r')) if os.path.exists(self.data_file) else {}
        data[os.path.basename(self.image_list[self.current_idx])] = {"ellipse": self.points_ellipse, "side": self.points_side}
        json.dump(data, open(self.data_file, 'w'), indent=4)
        QMessageBox.information(self, "저장", "좌표가 저장되었습니다.")

    def jump_to(self, t):
        if not self.image_list: return
        if t=="first": self.current_idx=0
        elif t=="last": self.current_idx=len(self.image_list)-1
        elif t=="prev": self.current_idx=max(0, self.current_idx-1)
        elif t=="next": self.current_idx=min(len(self.image_list)-1, self.current_idx+1)
        self.load_image()

    def change_mode(self): self.current_mode = "ellipse" if self.rb_ellipse.isChecked() else "side"

    def on_mouse_click(self, e):
        if not self.display_pixmap: return
        scale = min(self.img_display.width()/self.display_pixmap.width(), self.img_display.height()/self.display_pixmap.height())
        ox = (self.img_display.width() - self.display_pixmap.width()*scale)/2
        oy = (self.img_display.height() - self.display_pixmap.height()*scale)/2
        ix, iy = (e.x()-ox)/scale, (e.y()-oy)/scale
        if e.button() == Qt.LeftButton:
            (self.points_ellipse if self.current_mode=="ellipse" else self.points_side).append((ix, iy))
        elif e.button() == Qt.RightButton:
            pts = self.points_ellipse if self.current_mode=="ellipse" else self.points_side
            if pts: pts.pop()
        self.update_display()

    def closeEvent(self, event):
        """프로그램 종료 시 호출되는 이벤트"""
        # 최적화 모니터 창이 열려 있다면 닫기
        if hasattr(self, 'mon') and self.mon is not None:
            self.mon.close()
            self.mon = None
            
        # 설정값 등을 마지막으로 저장하고 싶다면 여기서 호출
        self.save_settings()
        
        # 종료 수락
        event.accept()

    def update_display(self):
        if self.display_pixmap is None: return
        
        canvas = self.display_pixmap.copy()
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. 클릭한 데이터 포인트 그리기
        painter.setPen(QPen(Qt.red, 10)); [painter.drawPoint(QPointF(*p)) for p in self.points_ellipse]
        painter.setPen(QPen(Qt.cyan, 10)); [painter.drawPoint(QPointF(*p)) for p in self.points_side]
        
        # 2. [핵심] 점이 5개 이상이면 실시간으로 타원 피팅 결과 표시
        if len(self.points_ellipse) >= 5:
            try:
                # OpenCV의 fitEllipse는 5개 이상의 점이 필요함
                pts_array = np.array(self.points_ellipse, dtype=np.float32)
                ellipse_geom = cv2.fitEllipse(pts_array) # (center, axes, angle)
                
                # 가이드 타원 스타일 (반투명 빨간색 점선)
                painter.setPen(QPen(QColor(255, 0, 0, 150), 2, Qt.DashLine))
                painter.save()
                
                # 타원 좌표계로 변환하여 그리기
                painter.translate(ellipse_geom[0][0], ellipse_geom[0][1])
                painter.rotate(ellipse_geom[2])
                
                # drawEllipse는 중심 기준이 아니라 좌상단 기준이므로 조정
                w, h = ellipse_geom[1][0], ellipse_geom[1][1]
                painter.drawEllipse(QRectF(-w/2, -h/2, w, h))
                
                painter.restore()
            except Exception as e:
                print(f"Ellipse fitting error: {e}")


        # 2. 캘리브레이션 결과 오버레이
        if self.is_overlay_on and self.last_calib_params is not None:
            rx, ry, rz, tx, ty, tz = self.last_calib_params
            R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
            t = np.array([tx, ty, tz])
            
            img_h, img_w = self.original_cv_img.shape[:2]
            f = float(self.focal_input.text()) if self.focal_input.text() else 3000
            #K = np.array([[f, 0, img_w/2], [0, f, img_h/2], [0, 0, 1]], dtype=np.float32)
            K = np.array([[f, 0, img_w/2], [0, f, img_h], [0, 0, 1]], dtype=np.float32)
            real_r = (float(self.diameter_input.text()) if self.diameter_input.text() else 100.0) / 2.0

            # --- [A] 3D 월드 좌표계 축 (XYZ) 및 문자 표시 ---
            # GUI에서 입력받은 축 길이 사용 (입력값이 없거나 오류 시 기본값 150)
            try:
                axis_len = float(self.axis_len_input.text())
            except:
                axis_len = 150.0

            pts_3d_axes = np.array([
                [0, 0, 0],         # 0: 원점
                [axis_len, 0, 0],   # 1: X축 끝
                [0, axis_len, 0],   # 2: Y축 끝
                [0, 0, axis_len]    # 3: Z축 끝
            ], dtype=np.float32).T

            pts_cam_axes = R @ pts_3d_axes + t.reshape(3, 1)
            pts_2d_axes = (K @ pts_cam_axes)
            pts_2d_axes = (pts_2d_axes[:2] / (pts_2d_axes[2] + 1e-9)).T
            
            origin = QPointF(*pts_2d_axes[0])
            labels = ["", "X", "Y", "Z"]
            colors = [None, Qt.red, Qt.green, Qt.cyan] # Blue는 어두워서 Cyan으로 변경 추천

            font = QFont('Arial', 14, QFont.Bold)
            painter.setFont(font)

            for i in range(1, 4):
                end_pt = QPointF(*pts_2d_axes[i])
                
                # 1. 축 선 그리기
                painter.setPen(QPen(colors[i], 3))
                painter.drawLine(origin, end_pt)
                
                # 2. 축 문자 표시 (가독성 강화)
                diff = end_pt - origin
                dist = np.sqrt(diff.x()**2 + diff.y()**2) + 1e-6
                offset_pt = end_pt + (diff / dist) * 25 # 글자를 선 끝에서 조금 더 띄움

                # 글자 외곽선 효과 (검은색 배경을 먼저 그려서 글자를 도드라지게 함)
                painter.setPen(QPen(Qt.black, 4)) # 외곽선 두께
                painter.drawText(offset_pt + QPointF(-5, 5), labels[i])
                
                # 실제 색상 글자 쓰기
                painter.setPen(QPen(colors[i]))
                painter.drawText(offset_pt + QPointF(-5, 5), labels[i])


            # --- [B] 원통 모델 (상단 타원 + 옆면 기둥) ---
            # 실시간으로 반영되는 real_r (지름/2)
            current_r = (float(self.dia_spin.value()) if self.dia_spin.value() else 100.0) / 2.0
            cylinder_height = float(self.height_spin.value()) if self.height_spin.value() else 100.0
            self.cyl_diameter = self.dia_spin.value()
            self.cyl_height = self.height_spin.value()

            angles = np.linspace(0, 2*np.pi, 100)
            # 상단 원 (z=0)
            top_circle = np.array([current_r*np.cos(angles), current_r*np.sin(angles), np.zeros(100)])
            # 하단 원 (z=height)
            bot_circle = np.array([current_r*np.cos(angles), current_r*np.sin(angles), np.full(100, cylinder_height)])
            
            # 투영 함수
            def project(pts_3d):
                pts_cam = R @ pts_3d + t.reshape(3, 1)
                pts_2d = (K @ pts_cam)
                return (pts_2d[:2] / (pts_2d[2] + 1e-9)).T

            pts_top = project(top_circle)
            pts_bot = project(bot_circle)

            def draw_high_contrast_line(p1, p2, is_dash=False):
                """검은색 두꺼운 선 위에 노란색 얇은 선을 겹쳐 그려 대비를 높임"""
                # 1. 배경 검은색 선 (외곽선 역할)
                pen_bg = QPen(Qt.black, 4, Qt.SolidLine if not is_dash else Qt.DashLine)
                painter.setPen(pen_bg)
                painter.drawLine(p1, p2)
                # 2. 전경 노란색 선
                pen_fg = QPen(QColor(255, 255, 0), 1.5, Qt.SolidLine if not is_dash else Qt.DashLine)
                painter.setPen(pen_fg)
                painter.drawLine(p1, p2)

            def draw_high_contrast_poly(pts, is_dash=False):
                """다각형(타원)에 검은 외곽선 적용"""
                poly = QPolygonF([QPointF(*p) for p in pts])
                # 검은색 외곽선
                painter.setPen(QPen(Qt.black, 4, Qt.SolidLine if not is_dash else Qt.DashLine))
                painter.drawPolygon(poly)
                # 노란색 메인선
                painter.setPen(QPen(QColor(255, 255, 0), 1.5, Qt.SolidLine if not is_dash else Qt.DashLine))
                painter.drawPolygon(poly)

            # 1. 상단 타원 (실선)
            draw_high_contrast_poly(pts_top, is_dash=False)
            
            # 2. 하단 타원 (밑면 - 점선/실선 선택 가능)
            # 밑면이 너무 안 보인다면 점선 대신 실선으로 바꾸는 것도 방법입니다.
            draw_high_contrast_poly(pts_bot, is_dash=True)

            # 3. 옆면 기둥 (4개의 연결선)
            for i in [0, 25, 50, 75]:
                draw_high_contrast_line(QPointF(*pts_top[i]), QPointF(*pts_bot[i]))


            # --- [B] 상단 타원 모델 (Yellow) ---
            angles = np.linspace(0, 2*np.pi, 100, endpoint=True) # 점 개수를 늘려 더 부드럽게
            circle_3d = np.array([real_r*np.cos(angles), real_r*np.sin(angles), np.zeros(100)])
            pts_2d_circle = (K @ (R @ circle_3d + t.reshape(3, 1)))
            pts_2d_circle = (pts_2d_circle[:2] / (pts_2d_circle[2] + 1e-9)).T
            
            # 펜 설정을 새로 고침 (검은색 외곽선 설정이 남지 않도록)
            painter.setPen(QPen(QColor(255, 255, 0, 180), 2, Qt.SolidLine)) 
            painter.setBrush(Qt.NoBrush) # 내부 채우기 없음
            
            poly = QPolygonF()
            for p in pts_2d_circle:
                poly.append(QPointF(p[0], p[1]))
            
            # Polyline보다는 Polygon이 끝점이 깔끔하게 닫힙니다.
            painter.drawPolygon(poly)

            # --- [C] 좌측 상단 캘리브레이션 정보 텍스트 ---
            painter.setPen(QPen(QColor(0, 0, 0, 160))) # 배경 박스
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRect(10, 10, 220//2, 140//2) # 
            
            painter.setPen(QPen(Qt.white))
            painter.setFont(QFont('Consolas', 5)) # default 10
            info_lines = [
                f"[ 6-DOF Calibration ]",
                f"Pitch(RX): {np.rad2deg(rx):8.2f} deg",
                f"Yaw  (RY): {np.rad2deg(ry):8.2f} deg",
                f"Roll (RZ): {np.rad2deg(rz):8.2f} deg",
                f"Pos  (TX): {tx:8.2f} mm",
                f"     (TY): {ty:8.2f} mm",
                f"Dist (TZ): {tz:8.2f} mm",
            ]
            for i, line in enumerate(info_lines):
                painter.drawText(20, 35//2 + (i * 18//2), line)

        painter.end()
        self.img_display.setPixmap(canvas.scaled(self.img_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def run_calibration(self):
        if len(self.points_ellipse) < 5 or len(self.points_side) < 2:
            QMessageBox.warning(self, "데이터 부족", "타원 5개, 측면 2개 이상이 필요합니다.")
            return

        # --- [1. 정밀 좌표계 구성] ---
        # 원본 이미지의 실제 해상도 기준 (GUI 표시 크기가 아님)
        img_h, img_w = self.original_cv_img.shape[:2]
        f_val = float(self.focal_input.text()) if self.focal_input.text() else 3000
        real_r = (float(self.diameter_input.text()) if self.diameter_input.text() else 100.0) / 2.0
        initial_distance = float(self.initial_dist_input.text()) if self.initial_dist_input.text() else 1500.0
        
        # 실제 이미지 중심점 (Principal Point)
        #K = np.array([[f_val, 0, img_w/2], [0, f_val, img_h/2], [0, 0, 1]], dtype=np.float32)
        K = np.array([[f_val, 0, img_w/2], [0, f_val, img_h], [0, 0, 1]], dtype=np.float32)
        
        # 점 데이터 (이미 원본 해상도 좌표로 저장되어 있어야 함)
        e_pts = np.array(self.points_ellipse, dtype=np.float32)
        s_pts = np.array(self.points_side, dtype=np.float32)

        # 초기 TX, TY 추정 (타원 중심을 이미지 중심으로부터의 오프셋으로 변환)
        obs_el = cv2.fitEllipse(e_pts)
        (cx_e, cy_e), (d1, d2), ang_e = obs_el
        estimated_pitch = np.arccos(np.clip(min(d1, d2)/max(d1, d2), 0.1, 1.0))

        # 역투영 공식: TX = (x - cx) * TZ / f
        init_tx = (cx_e - img_w/2) * initial_distance / f_val
        #init_ty = (cy_e - img_h/2) * initial_distance / f_val
        init_ty = (cy_e - img_h) * initial_distance / f_val

        A, B, C, D, E, F = self.get_ellipse_params(obs_el)

        a, b = d1 / 2, d2 / 2
        target_stat_area = a * b * np.pi

        def get_side_vec(pts):
            if len(pts) < 2: return None
            vx, vy, _, _ = cv2.fitLine(np.array(pts, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            d = np.array([vx[0], vy[0]], dtype=np.float32).flatten()
            if d[1] < 0: d = -d
            return d / np.linalg.norm(d)
        
        v1, v2 = get_side_vec(s_pts[:2]), get_side_vec(s_pts[2:4])

        self.mon = MonitorWindow(self.display_pixmap, e_pts, s_pts)
        self.mon.setAttribute(Qt.WA_DeleteOnClose) # 창이 닫힐 때 메모리 해제
        self.mon.show()

        # --- [2. 목적 함수] ---
        def objective(params):
            rx, ry, rz, tx, ty, tz = params
            R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
            t = np.array([tx, ty, tz])
            
            # 모델 투영
            angles = np.linspace(0, 2*np.pi, 40, endpoint=False)
            circle_3d = np.array([real_r*np.cos(angles), real_r*np.sin(angles), np.zeros(40)])
            pts_cam = R @ circle_3d + t.reshape(3, 1)
            
            # 카메라 뒤쪽 필터링 (강력한 페널티)
            if np.any(pts_cam[2,:] <= 10): return 1e12

            # 투영 및 분모 안전 처리P
            z_vals = pts_cam[2] + 1e-9
            pts_2hom = K @ pts_cam
            pts_2d = (pts_2hom[:2] / z_vals).T

            pts2d = K @ pts_cam
            x, y = pts2d[0]/pts2d[2], pts2d[1]/pts2d[2]

            # 에러 1: Taubin (정밀도)
            norm_factor = np.sqrt(A**2 + B**2 + C**2 + D**2 + E**2)
            f_val = (A*x**2 + B*x*y + C*y**2 + D*x + E*y + F) / (norm_factor + 1e-9)
            taubin_err = np.mean(f_val**2)

            # 에러 2: 면적 (스케일)
            proj_area = np.pi * np.sqrt(np.var(x) * np.var(y) * 4)
            area_err = (np.log(proj_area + 1e-6) - np.log(target_stat_area + 1e-6))**2

            # 에러 3: 수직축 정렬 (회전)
            z_axis_3d = np.array([0, 0, 1])
            z_axis_cam = R @ z_axis_3d
            z_proj = (K[:2, :2] @ z_axis_cam[:2])
            z_proj /= (np.linalg.norm(z_proj) + 1e-6)
            vert_err = (1.0 - np.abs(np.dot(z_proj, v1))) + (1.0 - np.abs(np.dot(z_proj, v2)))

            loss = taubin_err * 100 + area_err * 10000 + vert_err * 20000
            print(taubin_err* 100 , area_err* 10000, vert_err* 20000)

            # 시각화용 데이터
            ax_cam = R @ np.array([[0, 0], [0, 0], [0, 250]]) + t.reshape(3, 1)
            ax_2d = (K @ ax_cam); ax_2d = (ax_2d[:2]/ax_2d[2]).T
            self.mon.update_view(pts_2d, ax_2d, e_pts, s_pts, loss)
            return loss

        # --- [3. 초기값 및 범위 최적화] ---
        # 초기값: Pitch 45도, 추정된 TX/TY, 입력된 TZ
        initial_guess = [estimated_pitch, 0, 0, init_tx, init_ty, initial_distance]
        
        # 범위를 너무 크게 주지 않아 엉뚱한 해를 방지
        bounds = [
            #(0, np.pi/2),               # Pitch (45도 근처)
            #(-np.pi/6, np.pi/6), 
            (-np.pi/2, -np.pi/8),               # Pitch (45도 근처)
            (-np.pi/2, -np.pi/8), 
            (-0.01, 0.01),              # Yaw, Roll (미세 조정)
            # (init_tx-50, init_tx+50),   # TX (초기값 근처)
            # (init_ty-50, init_ty+50),   # TY (초기값 근처)
            (init_tx-50, init_tx+50),   # TX (초기값 근처)
            (init_ty-50, init_ty+50),   # TY (초기값 근처)
            (100, 500)                  # TZ
        ]

        res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-10, 'maxiter': 50000})
        self.show_results(res.x, res.fun)

        if res.success:
            self.last_calib_params = res.x # 최신 결과 저장
            # 최적화 끝나면 자동으로 오버레이 켜기 (선택 사항)
            self.is_overlay_on = True
            self.btn_overlay.setChecked(True)
            self.update_display()

    def get_ellipse_params(self, ellipse):
        """
        cv2.fitEllipse 결과(중심, 축, 각도)로부터 
        타원의 일반형 계수(A, B, C, D, E, F)를 추출합니다.
        Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        """
        (xc, yc), (d1, d2), angle = ellipse
        # a: 장축의 반, b: 단축의 반 (OpenCV는 d1, d2가 전체 길이임)
        a, b = d1 / 2.0, d2 / 2.0
        theta = np.deg2rad(angle)
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # 수학적 타원 방정식을 전개한 계수들
        A = (a * sin_t)**2 + (b * cos_t)**2
        B = 2 * (b**2 - a**2) * sin_t * cos_t
        C = (a * cos_t)**2 + (b * sin_t)**2
        D = -2 * A * xc - B * yc
        E = -B * xc - 2 * C * yc
        F = A * xc**2 + B * xc * yc + C * yc**2 - (a * b)**2
        
        return A, B, C, D, E, F
    
    def show_results(self, params, loss):
        """최적화된 6-DOF 파라미터를 사용자 친화적인 형식으로 출력"""
        rx, ry, rz, tx, ty, tz = params
        
        # 라디안을 도(degree)로 변환
        deg_x, deg_y, deg_z = np.rad2deg([rx, ry, rz])
        
        # 결과 문자열 구성
        result_text = (
            f"=== 6-DOF Optimization Result ===\n\n"
            f"[Translation - Distance]\n"
            f"- TZ (Distance): {tz:.2f} mm  <-- 핵심 측정값\n"
            f"- TX, TY: {tx:.2f}, {ty:.2f} mm\n\n"
            f"[Rotation - Euler Angles]\n"
            f"- Rx (Pitch): {deg_x:.2f}°\n"
            f"- Ry (Yaw): {deg_y:.2f}°\n"
            f"- Rz (Roll): {deg_z:.2f}°\n\n"
            f"[Optimization Info]\n"
            f"- Final Loss: {loss:.6e}\n"
            f"- Status: Success"
        )
        
        # 1. 메시지 박스로 즉시 알림
        QMessageBox.information(self, "최적화 완료", result_text)
        
        # 2. 콘솔에 상세 출력 (복사 용이)
        print("\n" + "="*30)
        print(result_text)
        print("="*30)

        # 3. (선택사항) 결과값 자동 저장
        self.save_result_to_json(params, loss)

    def save_result_to_json(self, params, loss):
        """계산된 6-DOF 결과를 별도의 파일로 기록"""
        result_file = "calibration_results_log.json"
        img_name = os.path.basename(self.image_list[self.current_idx]) if self.image_list else "unknown"
        
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image": img_name,
            "params": {
                "rx": params[0], "ry": params[1], "rz": params[2],
                "tx": params[3], "ty": params[4], "tz": params[5]
            },
            "loss": loss
        }
        
        log_data = []
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                try: log_data = json.load(f)
                except: log_data = []
        
        log_data.append(entry)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4)


if __name__ == '__main__':
    app = QApplication(sys.argv); ex = CalibratorGUI(); ex.show(); sys.exit(app.exec_())