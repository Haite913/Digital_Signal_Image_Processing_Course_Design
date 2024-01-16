# image_processor_ui.py
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QFileDialog, QMessageBox, QFrame, QLineEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

class ImageProcessorUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("图像处理")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        with open('style.qss', 'r') as file:
            self.setStyleSheet(file.read())

        main_layout = QHBoxLayout(self.central_widget)

        # 左侧布局（直方图均衡化和文字提交部分）
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.Box)
        left_layout = QVBoxLayout(left_frame)

        self.histogram_button = QPushButton("直方图均衡化")
        left_layout.addWidget(self.histogram_button)
        self.histogram_button.clicked.connect(self.histogram_equalization)

        self.text_input1 = QLineEdit(self)
        self.text_input1.setPlaceholderText("水印")
        left_layout.addWidget(self.text_input1)
        self.text_input1.returnPressed.connect(self.submit_water_mark) #输入回车

        self.curve_adjust_button = QPushButton("曲线调整")
        left_layout.addWidget(self.curve_adjust_button)
        self.curve_adjust_button.clicked.connect(self.curve_adjust)
    

        self.save_button = QPushButton("保存图像", self)
        left_layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_image)


        # 创建一个用于选择 ROI 的按钮
        self.roi_button = QPushButton("选择裁剪区域", self)
        left_layout.addWidget(self.roi_button)
        self.roi_button.clicked.connect(self.crop_image)

        main_layout.addWidget(left_frame, 1)

        # 中间布局（图像显示区域）
        middle_frame = QFrame()
        middle_frame.setFrameShape(QFrame.Box)
        middle_layout = QVBoxLayout(middle_frame)

        # 上半部分：图像显示区域
        self.load_image_button = QPushButton("从本地文件加载图像")
        self.load_image_button.clicked.connect(self.load_image)
        middle_layout.addWidget(self.load_image_button)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        middle_layout.addWidget(self.image_label)

        # 下半部分：旋转按钮区域
        rotate_buttons_layout = QHBoxLayout()

        self.left_rotate_button = QPushButton("左旋转")
        rotate_buttons_layout.addWidget(self.left_rotate_button)
        self.left_rotate_button.clicked.connect(self.rotate_left)

        self.right_rotate_button = QPushButton("右旋转")
        rotate_buttons_layout.addWidget(self.right_rotate_button)
        self.right_rotate_button.clicked.connect(self.rotate_right)

        middle_layout.addLayout(rotate_buttons_layout)  # 将旋转按钮区域添加到中间布局中

        main_layout.addWidget(middle_frame, 3)

        # 右侧布局（滑动按钮部分）
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.Box)
        right_layout = QVBoxLayout(right_frame)

        self.feature_labels = [
            QLabel("亮度"),
            QLabel("曝光"),
            QLabel("对比度"),
            QLabel("饱和度"),
            QLabel("HSL"),
            QLabel("锐化"),
            QLabel("平滑"),
            QLabel("色温"),
            QLabel("色调调节")
        ]

        self.sliders = []
        for label in self.feature_labels:
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)
            slider.setVisible(True)
            slider.valueChanged.connect(self.adjust_value)
            
            if label.text() == "锐化":
                slider.setMinimum(0)
            elif label.text() == "平滑":
                slider.setMinimum(0)
            
                
            right_layout.addWidget(label)
            right_layout.addWidget(slider)
            self.sliders.append(slider)

        # 添加文本输入框
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("文字提交：输入文字")
        self.text_input.returnPressed.connect(self.submit_text) #输入回车
        right_layout.addWidget(self.text_input)

         


        main_layout.addWidget(right_frame, 1)
