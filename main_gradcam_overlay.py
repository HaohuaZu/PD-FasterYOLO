import os
import random
import time
import argparse
import numpy as np
from models.gradcam import YOLOV5GradCAM, YOLOV5GradCAMPP
from models.yolov5_object_detector import YOLOV5TorchObjectDetector
import cv2

names = ['bridge', 'missing_component', 'missing_solder']
target_layers = ['model_17_cv3_act']  #, 'model_20_cv3_act', 'model_23_cv3_act'

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str,
                    default=r"D:\PCB_defect\runs\train\yolov5n_PCB_distill_exp64\weights\best.pt",
                    help='Path to the model')
parser.add_argument('--img-path', type=str, default=r'D:\PCB_defect\datasets\Joint_PCB\Images\41_1.bmp',
                    help='input image path')
parser.add_argument('--output-dir', type=str, default='outputs/', help='output dir')
parser.add_argument('--img-size', type=int, default=640, help="input image size")
parser.add_argument('--target-layer', type=str, default='model_17_cv3_act',
                    help='The layer hierarchical address to which gradcam will applied')
parser.add_argument('--method', type=str, default='gradcam', help='gradcam method')
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--no_text_box', action='store_true',
                    help='do not show label and box on the heatmap')
args = parser.parse_args()


def get_res_img(bbox, mask, input_size):
    # 处理 mask，避免无效值
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    mask = np.nan_to_num(mask)  # 将 NaN 转换为 0
    mask = mask.astype(np.uint8)

    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    n_heatmat = (heatmap / 255).astype(np.float32)

    # 调整热力图大小以匹配输入尺寸
    if n_heatmat.shape[:2] != (input_size[0], input_size[1]):
        n_heatmat = cv2.resize(n_heatmat, (input_size[1], input_size[0]))

    return n_heatmat


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # 确保图像是正确的格式
    img_uint8 = (img * 255).astype(np.uint8)

    # 绘制边界框和标签
    # tl = line_thickness or round(0.002 * (img_uint8.shape[0] + img_uint8.shape[1]) / 2) + 1
    # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(img_uint8, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    #
    # if label:
    #     tf = max(tl - 1, 1)
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    #     cv2.rectangle(img_uint8, c1, c2, color, -1, cv2.LINE_AA)
    #     cv2.putText(img_uint8, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
    #                 lineType=cv2.LINE_AA)

    return img_uint8


def main(img_path):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    device = args.device
    input_size = (args.img_size, args.img_size)

    img = cv2.imread(img_path)
    print('[INFO] Loading the model')
    model = YOLOV5TorchObjectDetector(args.model_path, device, img_size=input_size, names=names)
    torch_img = model.preprocessing(img[..., ::-1])
    tic = time.time()

    # 创建一个累积热力图
    accumulated_heatmap = np.zeros((input_size[0], input_size[1], 3), dtype=np.float32)
    # 创建一个列表存储所有检测框和标签信息
    all_boxes = []
    all_labels = []
    all_colors = []

    # 获取原始图像
    result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    result = result[..., ::-1]  # BGR
    result = cv2.resize(result, (input_size[1], input_size[0]))  # 确保尺寸匹配
    result = result.astype(np.float32) / 255.0  # 归一化到 [0,1]

    # 遍历所有检测层
    for target_layer in target_layers:
        if args.method == 'gradcam':
            saliency_method = YOLOV5GradCAM(model=model, layer_name=target_layer, img_size=input_size)
        elif args.method == 'gradcampp':
            saliency_method = YOLOV5GradCAMPP(model=model, layer_name=target_layer, img_size=input_size)

        masks, logits, [boxes, _, class_names, conf] = saliency_method(torch_img)

        # 处理该层的所有目标
        for i, mask in enumerate(masks):
            heatmap = get_res_img(boxes[0][i], mask, input_size)
            accumulated_heatmap += heatmap

            # 存储框和标签信息
            bbox, cls_name = boxes[0][i], class_names[0][i]
            label = f'{cls_name} {conf[0][i]:.2f}'
            all_boxes.append(bbox)
            all_labels.append(label)
            all_colors.append(colors[int(names.index(cls_name))])

    # 归一化累积热力图
    if accumulated_heatmap.max() > 0:
        accumulated_heatmap = accumulated_heatmap / accumulated_heatmap.max()

    # 将热力图叠加到原始图像上
    final_img = cv2.addWeighted(result, 0.3, accumulated_heatmap, 0.7, 0)

    # 添加所有边界框和标签
    for bbox, label, color in zip(all_boxes, all_labels, all_colors):
        final_img = plot_one_box(bbox, final_img, label=label, color=color, line_thickness=3)

    # 保存结果
    imgae_name = os.path.basename(img_path)
    save_path = f'{args.output_dir}{imgae_name[:-4]}/{args.method}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 缩放到原图片大小
    final_img = cv2.resize(final_img, (img.shape[1], img.shape[0]))
    output_path = f'{save_path}/combined_heatmap.jpg'
    cv2.imwrite(output_path, final_img)
    print(f'Combined heatmap saved to {output_path}')
    print(f'Total time : {round(time.time() - tic, 4)} s')


if __name__ == '__main__':
    if os.path.isdir(args.img_path):
        img_list = os.listdir(args.img_path)
        print(img_list)
        for item in img_list:
            main(os.path.join(args.img_path, item))
    else:
        main(args.img_path)