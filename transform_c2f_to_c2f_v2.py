import torch
from models.yolo import replace_c2f_with_c2f_v2, attempt_load

if __name__ == '__main__':
    model = torch.load('runs/train/yolov5n_mobilenetv3_c2f_Faster/weights/best.pt')
    if model['ema']:
        model = model['ema'].float()
    else:
        model = model['model'].float()
    
    model = model.cpu()
    replace_c2f_with_c2f_v2(model)
    torch.save({'model':model.half(), 'ema':None}, 'transform_model.pt')
    