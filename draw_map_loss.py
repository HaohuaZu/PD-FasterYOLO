import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # 列出待获取数据内容的文件位置
    # v5、v8都是csv格式的，v7是txt格式的
    result_dict = {
        'YOLOv5n': r'D:\PCB_defect\RTDETR-main\runs\train\exp6\results.csv',
        'FasterYOLO': r'D:\PCB_defect\code_YOLO\yolov5-prune-20240303\yolov5-prune-main\runs\train\exp56\results.csv',
        #'FasterYOLO-PD': r'D:\PCB_defect\runs\train\yolov5n_PCB_distill_exp65\results.csv',
        #'YOLOv5n-PD': r'D:\PCB_defect\runs\train\yolov5n_PCB_distill_exp57\results.csv'
    }

    # 绘制map50
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[6]).values.ravel()    # 6是指map50的下标（每行从0开始向右数）
        else:   # 文件后缀是txt
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:
                    data.append(float(d.strip().split()[10]))   # 10是指map50的下标（每行从0开始向右数）
                data = np.array(data)
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='1')   # 线条粗细设为1

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5')
    plt.legend()
    plt.grid()
    # 显示图像
    plt.savefig("mAP50.png", dpi=600)   # dpi可设为300/600/900，表示存为更高清的矢量图
    plt.show()


    # 绘制map50-95
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[7]).values.ravel()    # 7是指map50-95的下标（每行从0开始向右数）
        else:
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:
                    data.append(float(d.strip().split()[11]))   # 11是指map50-95的下标（每行从0开始向右数）
                data = np.array(data)
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='1')

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5:0.95')
    plt.legend()
    plt.grid()
    # 显示图像
    plt.savefig("mAP50-95.png", dpi=600)
    plt.show()

    # 绘制训练的总loss
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            box_loss = pd.read_csv(res_path, usecols=[1]).values.ravel()
            obj_loss = pd.read_csv(res_path, usecols=[2]).values.ravel()
            cls_loss = pd.read_csv(res_path, usecols=[3]).values.ravel()
            data = np.round(box_loss + obj_loss + cls_loss, 5)    # 3个loss相加并且保留小数点后5位（与v7一致）

        else:
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:
                    data.append(float(d.strip().split()[5]))
                data = np.array(data)
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='1')

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    # 显示图像
    plt.savefig("loss.png", dpi=600)
    plt.show()
