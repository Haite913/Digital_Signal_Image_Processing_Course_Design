# image_processor.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, \
    QFileDialog, QMessageBox, QFrame,QLineEdit,QDialog,QApplication, QGraphicsView, QGraphicsScene, QGraphicsLineItem
from PyQt5.QtCore import Qt,QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
import cv2
import numpy as np
from image_processor_ui import ImageProcessorUI
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy.interpolate as inter
import numpy as np
import subprocess

class ImageProcessor(ImageProcessorUI):

    def __init__(self):
        super().__init__()

        # 从 UI 文件中继承 GUI 部分的代码
        self.current_image = None
        self.gamma = 1.0 
        #用于标记，判断是否要保存当前图片，用于同时调整参数
        
        # 定义操作变量，判断当前操作是否与上一个操作不同
        self.global_counter = -1
        # 用于临时保存图片，当多参数调节时使用
        self.global_image = None

        # 定义
        self.equalization_counter = 1
        self.equalization_image = None
        
    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)",
                                                    options=options)
        if file_path:
            self.current_image = cv2.imread(file_path)  # 使用OpenCV读取图像文件
            if self.current_image is None:
                QMessageBox.critical(self, "错误", "无法加载图像文件")
            else:
                # 将图像转换为RGB格式
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                self.display_image(self.current_image)
            self.global_image=self.current_image

    def display_image(self, image):
        # 获取原始图像的高度和宽度
        height, width, channel = image.shape
        # 计算显示图像的高度和宽度（可根据需要调整比例）
        display_height = 400
        display_width = int((width / height) * display_height)
        # 调整图像大小并显示
        resized_image = cv2.resize(image, (display_width, display_height))
        bytes_per_line = 3 * display_width
        q_image = QImage(resized_image.data, display_width, display_height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def adjust_value(self, value):
        sender = self.sender()
        index = self.sliders.index(sender)
        if self.current_image is not None:
            if index == 0:
                print(self.global_counter)
                if self.global_counter != 0 or self.global_counter==-1:
                    print("保存上一次操作图")
                    self.current_image = self.global_image
                self.global_counter = 0
                self.adjust_brightness(value)  # 调节亮度
            elif index == 1:
                print(self.global_counter)
                if self.global_counter != 1 or self.global_counter==-1:
                    print("保存上一次操作图")
                    self.current_image = self.global_image
                self.global_counter = 1
                self.adjust_exposure(value)  # 调节曝光
            elif index == 2:
                print(self.global_counter)
                if self.global_counter != 2 or self.global_counter==-1:
                    print("保存上一次操作图")
                    self.current_image = self.global_image
                self.global_counter = 2
                self.adjust_contrast(value)  # 调节对比度
            elif index == 3:
                print(self.global_counter)
                if self.global_counter != 3 or self.global_counter==-1:
                    print("保存上一次操作图")
                    self.current_image = self.global_image
                self.global_counter = 3
                self.adjust_saturation(value)  # 调节饱和度
            elif index == 4:
                print(self.global_counter)
                if self.global_counter != 4 or self.global_counter==-1:
                    print("保存上一次操作图")
                    self.current_image = self.global_image
                self.global_counter = 4
                self.adjust_hsl(value)  # 调节HSL
            elif index == 5:
                print(self.global_counter)
                if self.global_counter != 5 or self.global_counter==-1:
                    print("保存上一次操作图")
                    self.current_image = self.global_image
                self.global_counter = 5
                self.adjust_sharpness(value)  # 调节锐化
            elif index == 6:
                print(self.global_counter)
                if self.global_counter != 6 or self.global_counter==-1:
                    print("保存上一次操作图")
                    self.current_image = self.global_image
                self.global_counter = 6
                self.adjust_smoothness(value)  # 调节平滑
            elif index == 7:
                print(self.global_counter)
                if self.global_counter != 7 or self.global_counter==-1:
                    print("保存上一次操作图")
                    self.current_image = self.global_image
                self.global_counter = 7
                self.adjust_color_temperature(value)  # 调节色温
            elif index == 8:
                print(self.global_counter)
                if self.global_counter != 8 or self.global_counter==-1:
                    print("保存上一次操作图")
                    self.current_image = self.global_image
                self.global_counter = 8
                self.adjust_hue(value)  # 调节色调


    def adjust_sensitivity(self, value):
        pass  # 调节光感

    def adjust_brightness(self, value):
        if self.current_image is not None:
            global_image = cv2.convertScaleAbs(self.current_image, alpha=1, beta=value)
            self.display_image(global_image)  # 展示调整后的图像
        pass  # 调节亮度

    def adjust_exposure(self, value):
        normalized_value = (value + 100) / 200  # 将值映射到0到1之间

        # 计算曝光调节系数
        exposure_factor = 2 ** normalized_value

        # 将图像转换为float32类型进行调节
        image_float = self.current_image.astype(np.float32) / 255.0

        # 应用曝光调节
        adjusted_image = exposure_factor * image_float
        adjusted_image = np.clip(adjusted_image, 0, 1)  # 将值限制在有效范围内
        self.global_image = (255 * adjusted_image).astype(np.uint8)  # 转换回uint8格式

        self.display_image(self.global_image)  # 显示调整后的图像
        pass  # 调节曝光

    def adjust_contrast(self, value):
        if self.current_image is not None:
            # 将value映射到[0.5, 1.5]范围
            value = 1.0 * (value + 200) / 200
    
            # 调整对比度
            adjusted_image = self.current_image.astype(float) * value
    
            # 将像素值限制在[0, 255]范围内
            adjusted_image = np.clip(adjusted_image, 0, 255)
    
            # 转换为无符号整型数组
            adjusted_image = adjusted_image.astype(np.uint8)
    
            # 更新全局图像
            self.global_image = adjusted_image
    
            # 展示调整后的图像
            self.display_image(self.global_image)
        pass  # 调节对比度
    
    def adjust_saturation(self, value):
        if self.current_image is not None:
            # 将图像转换为HSV颜色空间
            hsv_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2HSV)

            # 获取饱和度通道
            saturation_channel = hsv_image[:, :, 1]

            # 将范围从[-100, 100]映射到[0, 2]
            saturation_factor = (100 + value) / 100

            # 调整饱和度通道
            adjusted_saturation_channel = np.clip(saturation_channel * saturation_factor, 0, 255)

            # 将调整后的饱和度通道赋值回HSV图像
            hsv_image[:, :, 1] = adjusted_saturation_channel

            # 将图像转换回RGB颜色空间
            self.global_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

            # 显示调整后的图像
            self.display_image(self.global_image)
        pass  # 调节饱和度

    def adjust_hsl(self, value):
        if self.current_image is not None:
            # 将图像转换为HSL颜色空间
            hls_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2HLS)
            
            value /= 10

            # 调节HSL通道
            hls_image[..., 0] = (hls_image[..., 0] + value) % 180  # 调节色相
            hls_image[..., 1] = np.clip(hls_image[..., 1] * (1 + value / 100), 0, 255)  # 调节饱和度
            hls_image[..., 2] = np.clip(hls_image[..., 2] * (1 + value / 100), 0, 255)  # 调节亮度
            
            # 将图像转换回RGB颜色空间
            self.global_image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2RGB)
            
            # 显示调整后的图像
            self.display_image(self.global_image)
        pass  # 调节HSL

     #调节锐化
    def adjust_sharpness(self, value):
        if self.current_image is not None:
        # 范围从 [0, 100] 映射到 [0.0, 3.0]
            sharpening_factor = value / 100.0 * 3.0
            
            # 定义锐化卷积核--高波滤通
            kernel = np.array([[-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]])
            
            # 锐化图像
            self.global_image = cv2.filter2D(self.current_image, -1, kernel * sharpening_factor)
            
            # 如果滑块值为0，则显示原始图像
            if value == 0:
                self.global_image = self.current_image
        
        self.display_image(self.global_image) # 展示调整后的图像
    pass
    
    # 调节平滑
    def adjust_smoothness(self, value):
        if self.current_image is not None:
            # 范围从 [-100, 100] 映射到 [0, 10]
            smoothing_factor = (value) / 100.0 * 10.0

            # 定义高斯滤波器大小（奇数）
            kernel_size = int(smoothing_factor) * 2 + 1

            # 平滑图像
            self.global_image = cv2.GaussianBlur(self.current_image, (kernel_size, kernel_size), 0)        
            self.display_image(self.global_image) # 展示调整后的图像
        pass
    # 调节色温(有问题)
    def adjust_color_temperature(self, value):
        if self.current_image is not None:
            # 将范围从 [-100, 100] 映射到 [-180, 180]（调整的色温范围）
            temperature = value / 100.0 * 18.0            
            # 将图像从BGR转换为RGB
            image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)            
            # 将图像分离为红、绿、蓝三个通道
            r, g, b = cv2.split(image_rgb)            
            # 分别调整每个通道的色温
            r = cv2.addWeighted(r, 1.0, np.zeros_like(r), 0.0, -temperature)
            b = cv2.addWeighted(b, 1.0, np.zeros_like(b), 0.0, temperature)            
            # 将调整后的通道合并为RGB图像
            adjusted_image = cv2.merge([r, g, b])            
            # 将图像从RGB转换回BGR
            self.global_image = cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2BGR)            
            self.display_image(self.global_image)
    pass

   # 调节色调
    def adjust_hue(self, value):
        if self.current_image is not None:
             # 将图像从BGR转换为HSV
             hsv_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2HSV)            
             # 调整色调值
             hue_value = int(value*0.2) # 范围从 [-100, 100] 映射到 [-360, 360]（调整的色调范围） -180到180为HSV色彩空间中的完整色调周期，所以-360和360又回到初始 
             hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_value) % 180            
             # 将图像从HSV转换回BGR
             self.global_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)            
             # # 将原始图像与调整后的图像进行混合
             # blended_image = cv2.addWeighted(self.current_image, 0.5, adjusted_image, 0.5, 0)         
             self.display_image(self.global_image) # 展示调整后的图像

    def histogram_equalization(self):
        # 实现直方图均衡化的函数
        if self.current_image is not None:
            if(self.equalization_counter%2==1):
                self.equalization_image=self.current_image
                # 将图像转换为灰度图像
                gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
                
                # 进行直方图均衡化
                equalized_image = cv2.equalizeHist(gray_image)
                
                # 将均衡化后的图像转换回RGB颜色空间
                self.global_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)
                
                # 显示均衡化后的图像
                self.display_image(self.global_image)
            else:
                self.current_image=self.equalization_image
                # 显示均衡化后的图像
                self.display_image(self.current_image)
            
            self.equalization_counter+=1
    pass

    def rotate_left(self):
        # 左旋转
        if self.current_image is not None:
            if self.global_counter != 10 or self.global_counter==-1:
                print("保存上一次操作图")
                self.current_image = self.global_image
            self.global_counter = 10

            self.global_image = cv2.rotate(self.global_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.display_image(self.global_image)
        pass

    def rotate_right(self):
        # 右旋转
        if self.current_image is not None:
            if self.global_counter != 11 or self.global_counter==-1:
                print("保存上一次操作图")
                self.current_image = self.global_image
            self.global_counter = 11

            self.global_image = cv2.rotate(self.global_image, cv2.ROTATE_90_CLOCKWISE)
            self.display_image(self.global_image)
        pass

    # 文字提交
    def submit_text(self):
        if self.current_image is not None:
            # 获取输入的文本
            text = self.text_input.text()       
            # 创建一个副本，以免修改原始图像
            self.global_image = np.copy(self.current_image)
            # 在图像上绘制文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3  #文字的缩放比例
            font_thickness = 3 #文字的轮廓厚度
            color = (255, 255, 255) # 白色
            # 在图像中央绘制文本
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_position = ((self.global_image.shape[1] - text_size[0]) // 2, (self.global_image.shape[0] + text_size[1]) // 2)
            cv2.putText(self.global_image, text, text_position, font, font_scale, color, font_thickness)
            # 显示带有文本的图像
            self.display_image(self.global_image)
    pass

    def open_curve_adjustment_window(self):
        # 创建一个新的窗口
        curve_adjustment_window = CurveDragView()

    
    def crop_image(self):
        if self.current_image is not None:
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            # 使用 selectROI 获取用户选择区域的坐标
            roi = cv2.selectROI("Select Region of Interest", rgb_image,False,False)

            # 从图像中裁剪所选择的区域并保持数据类型和颜色空间一致
            cropped_image = self.current_image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])].copy()
            self.global_image = cropped_image.copy()
            # 显示裁剪后的图像
            self.display_image(self.global_image)
    
    def save_image(self):
        if self.current_image is not None:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "保存图像文件", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)",
                                                       options=options)
            if file_path:
                # 使用OpenCV保存图像到指定路径
                cv2.imwrite(file_path, cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
                QMessageBox.information(self, "保存成功", "图像保存成功")
                

    #添加水印
    def submit_water_mark(self):
        if self.current_image is not None:

            # 获取输入的文本
            text = self.text_input1.text()

            # 设定水印参数
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2  #文字的大小
            font_thickness = 10 #文字的轮廓厚度
            font_color = (255, 255, 255) # 白色


            # 设置文本文字(水印)的参数
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            text_position_1 = (10, self.current_image.shape[0] - 10)
            text_position_2 = ((self.current_image.shape[1] - text_size[0]) // 2, (self.current_image.shape[0] + text_size[1]) // 2)
            text_position_3 = (self.current_image.shape[1] - text_size[0] - 10, 100)

            # 创建一个空白图像，大小与原图相同
            text_image = np.zeros_like(self.current_image)

            # 在空白图像上添加文本
            cv2.putText(text_image, text, text_position_2, font, font_scale, font_color, font_thickness)
            cv2.putText(text_image, text, text_position_3, font, font_scale, font_color, font_thickness)
            cv2.putText(text_image, text, text_position_1, font, font_scale, font_color, font_thickness)

            # 获取旋转矩阵
            angle = 20
            rotation_matrix = cv2.getRotationMatrix2D((text_image.shape[1] // 2, text_image.shape[0] // 2), angle, 1)

            # 进行仿射变换
            rotated_text_image = cv2.warpAffine(text_image, rotation_matrix, (text_image.shape[1], text_image.shape[0]))

            # 叠加到原图上,并且让字体透明
            opacity = 0.2 #透明度值
            self.global_image = cv2.addWeighted(self.current_image, 1, rotated_text_image, opacity, 0)

            # 显示带有文本的图像
            self.display_image(self.global_image)
    
    def curve_adjust(temp1,temp2):
        subprocess.run(["python", "./curve.py"], check=True)

def main():
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
