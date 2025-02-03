import cv2
import numpy as np

# 定义类别名称和对应的BGR颜色
class_names = {0: "bridge", 1: "missing_component", 2: "missing_solder"}
colors = {1: (0x38, 0x38, 0xFF), 2: (0x97, 0x9D, 0xFF), 0: (0x1F, 0x70, 0xFF)}
# 使用示例
image_path = 'D:/PCB_defect/heatmap/92_2/92_2.bmp'  # 请替换为你的图像文件路径
label_file_path = r'D:\PCB_defect\heatmap\92_2\92_2.txt'  # 请替换为你的标注文件路径
output_image_path = r'D:\PCB_defect\heatmap\92_2\groundtruth.png'  # 输出图像路径

image = cv2.imread(image_path)

# 读取YOLO标注文件并解析
def read_yolo_labels(label_file_path):
    boxes = []
    with open(label_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            boxes.append([class_id, x_center, y_center, width, height])
    return boxes

# 在图像上绘制真实边界框
def draw_true_boxes(image, boxes, output_path, line_thickness=4):
    height, width, _ = image.shape
    for box in boxes:
        class_id, x_center, y_center, box_width, box_height = box
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)
        color = colors[int(class_id)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        class_name = class_names[int(class_id)]
        background_color = [int(c * 255) for c in color]  # 将BGR颜色转换为RGB
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (153, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(output_path, image)


boxes = read_yolo_labels(label_file_path)
draw_true_boxes(image, boxes, output_image_path)