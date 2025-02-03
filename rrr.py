import matplotlib.pyplot as plt

# 定义数据
models = ["YOLOv3", "YOLOv3-tiny", "YOLOv5-s", "YOLOv5-n", "YOLOv8-n", "YOLOv8-s", "GCC-YOLO", "transformer-YOLO",'FasterYOLO','YOLOv11']
map_values = [90.9, 91.3, 92.8, 90.2, 89.3, 90.8, 91.8,88.5,92.3,90.5]  # mAP 百分比转换为小数
fps_values = [65.8, 114.4, 264.8, 581.9, 426.7, 189.8, 85.3,49.7,595.7,514.2]  # 推理时间的倒数，即FPS
size_values = [92.8, 42.2, 14.4, 3.9, 92.8, 22.5, 4.2, 75.6,  3.4,5.5]  # 模型大小MB，用作半径
colors = ['blue', 'green', 'red', 'teal', 'magenta', 'olive', 'purple', 'orange', 'gray', 'brown']  # 为每个模型分配颜色

# 计算推理时间的倒数作为横坐标（推理速度）
inference_times = [1 / fps for fps in fps_values]

# 创建散点图
plt.figure(figsize=(10, 8))

# 在每个散点的右上角添加模型的名称
for i, model in enumerate(models):
    radius = size_values[i]  # 将模型大小用作半径
    plt.scatter(inference_times[i], map_values[i],
                s=radius * 10,  # 将半径放大以便在图上可见，这里乘以10是为了更好地展示大小差异
                color=colors[i],
                alpha=0.5)  # 设置透明度为0.5
    # 添加模型名称
    # plt.text(inference_times[i], map_values[i], model,
    #          horizontalalignment='left',
    #          verticalalignment='bottom',
    #          fontsize=12,
    #          color='black')

# 添加标签和标题
plt.xlabel('Inference Time (1/FPS)')
plt.ylabel('Mean Average Precision (mAP)')
plt.title('Visualization of Object Detection Models')

# 隐藏网格线
plt.grid(False)

# 显示图表
plt.show()