import matplotlib.pyplot as plt
import pandas as pd

# 绘制PR
def plot_PR():
    pr_csv_dict = {
        'YOLOv5-n': r'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\val\yolov5n5\PR_curve.csv',
        'FasterYOLO': r'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\val\FasterYOLO\PR_curve.csv',
        #'YOLOv5-s': r'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\val\yolov5s\PR_curve.csv',
        #'pd-v5': r'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\val\pd-v5\PR_curve.csv',
        #'PD-FasterYOLO': r'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\val\PD-FasterYOLO\PR_curve.csv',
    }

    # 绘制pr
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname in pr_csv_dict:
        res_path = pr_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[4]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth='2')

    # 添加x轴和y轴标签
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid()  # 显示网格线
    # 显示图像
    fig.savefig("pr.png", dpi=250)
    plt.show()

# 绘制F1
def plot_F1():
    f1_csv_dict = {
        'YOLOv5-n': r'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\val\yolov5n5\F1_curve.csv',
        'FasterYOLO': r'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\val\FasterYOLO\F1_curve.csv',
        #'YOLOv5-s': r'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\val\yolov5s\F1_curve.csv',
        #'pd-v5': r'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\val\pd-v5\F1_curve.csv',
        #'PD-FasterYOLO': r'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\val\PD-FasterYOLO\F1_curve.csv',
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname in f1_csv_dict:
        res_path = f1_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[5]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth='2')

    # 添加x轴和y轴标签
    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid()  # 显示网格线
    # 显示图像
    fig.savefig("F1.png", dpi=250)
    plt.show()

if __name__ == '__main__':
    plot_PR()   # 绘制PR
    plot_F1()   # 绘制F1
