import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'

# 初始化坐标点的初始位置
points = [(0, 0), (50, 50), (100, 100), (150, 150), (200, 200), (255, 255)]

# 创建一个画布和坐标系子图和图片子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_xlim(0, 255)  # 设置x轴范围
ax1.set_title('曲线调色(RED):')
ax1.set_ylim(0, 255)  # 设置y轴范围

# 加载并显示本地图片
image = mpimg.imread('./image/blank.jpeg')
ax2.imshow(image)
ax2.set_title('调色后的图片:')

# 绘制初始的六个点
scatter = ax1.scatter([point[0] for point in points], [point[1] for point in points], color='red')

# 提取坐标点的X和Y值
x = [point[0] for point in points]
y = [point[1] for point in points]

# 进行B样条曲线拟合
tck = splrep(x, y, k=3)  # k表示曲线的阶数
x_range = np.linspace(min(x), max(x), 100)
y_range = splev(x_range, tck)

# 绘制拟合的曲线
ax1.plot(x_range, y_range, color='blue')

def apply_curve_to_image(image, curve):
    # 获取图像的红色通道
    red_channel = image[:, :, 0]

    # 将曲线的长度插值为与红色通道相同的长度
    interpolated_curve = np.interp(np.arange(256), np.linspace(0, 255, len(curve)), curve)

    # 将插值后的曲线应用于红色通道
    adjusted_red_channel = interpolated_curve[red_channel.astype(int)]

    # 创建一个新的图像，将调整后的红色通道与原图像的绿色和蓝色通道组合起来
    adjusted_image = np.dstack((adjusted_red_channel, image[:, :, 1], image[:, :, 2]))

    # 将图像数据归一化到 [0, 1] 范围内
    adjusted_image = adjusted_image / 255.0

    return adjusted_image

def update_curve(event):
    global points, scatter, x, y, tck, x_range, y_range
    
    # Check if it's a left-click drag event and a scatter point is being dragged
    if event.button == 1 and scatter.contains(event)[0]:
        # Get the index of the dragged point
        ind = scatter.contains(event)[1]['ind'][0]
        
        # Calculate the new position of the point, considering the boundary constraints
        new_x = min(max(event.xdata, 0), 255)
        new_y = min(max(event.ydata, 0), 255)
        
        # Update the position of the dragged point in the points list
        points[ind] = (new_x, new_y)

        # Update the position of the dragged point in the scatter plot
        scatter.set_offsets(points)

        # Update the B-spline curve fit
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        tck = splrep(x, y, k=3)
        x_range = np.linspace(min(x), max(x), 100)
        y_range = splev(x_range, tck)
        ax1.lines[-1].set_xdata(x_range)
        ax1.lines[-1].set_ydata(y_range)

        # Apply the updated curve to the image and display it
        adjusted_image = apply_curve_to_image(image, y_range)
        ax2.imshow(adjusted_image)

        fig.canvas.draw()

# 注册回调函数
fig.canvas.mpl_connect('motion_notify_event', update_curve)

# 显示坐标系和图片
plt.show()