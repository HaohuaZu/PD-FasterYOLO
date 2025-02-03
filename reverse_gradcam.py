import cv2
import numpy as np

# 假设 final_img 是叠加后的图像（形状为 [H, W, 3]，值范围 [0, 255]）
final_img = cv2.imread(r'D:\PCB_defect\detimg_heatmap\92_2\featuremap\pd-v5.jpg')  # 加载已生成的热力图图像

# 反转颜色
final_img = 255 - final_img

# 保存反转后的图像
cv2.imwrite(r'D:\PCB_defect\detimg_heatmap\92_2\featuremap\pd-v5.jpg', final_img)