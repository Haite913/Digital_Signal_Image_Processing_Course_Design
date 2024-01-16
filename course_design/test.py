import numpy as np  
import matplotlib.pyplot as plt  
import scipy.signal as signal  
import cv2  
  
# 数字信号处理示例：肌肉活动信号处理  
def process_muscle_signal(data):  
    # 滤波器设计（示例使用带阻滤波器）  
    b, a = signal.iirnotch(10, 50)  
    filtered_data = signal.lfilter(b, a, data)  
      
    # 快速傅里叶变换（FFT）  
    frequencies, amplitudes = signal.periodogram(filtered_data, fs=1000)  # fs为采样频率  
      
    # 绘制频谱图  
    plt.plot(frequencies, amplitudes)  
    plt.xlabel('Frequency (Hz)')  
    plt.ylabel('Amplitude')  
    plt.title('Muscle Activity Spectrum')  
    plt.show()  
  
# 图像处理示例：骨骼形态分析  
def process_bone_image(image):  
    # 灰度化处理（如果图像是彩色的）  
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
      
    # 二值化处理（设置阈值）  
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)  
      
    # 边缘检测（示例使用Canny算法）  
    edges = cv2.Canny(binary_image, 50, 150)  
      
    # 绘制边缘检测结果  
    plt.imshow(edges, cmap='gray')  
    plt.title('Bone Edges')  
    plt.show()  
  
# 模拟数据生成（仅用于示例）  
# 在实际应用中，需要从传感器或图像采集设备获取真实数据  
muscle_activity = np.random.normal(0, 1, 1000)  # 模拟肌肉活动信号  
bone_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)  # 模拟骨骼形态图像  
  
# 处理肌肉活动信号并显示频谱图  
process_muscle_signal(muscle_activity)  
  
# 处理骨骼形态图像并显示边缘检测结果  
process_bone_image(bone_image)