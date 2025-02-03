import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import ComputeLoss
from utils.metrics import bbox_iou
from utils.general import check_version

# ------------------------------BiliBili 魔鬼面具up主------------------------------
class LogicalLoss(nn.Module):
    def __init__(self, hyp, model, distiller='l2') -> None:
        super().__init__()
        
        if distiller in ['l2', 'l1']:
            self.logical_loss = OutputLoss(hyp, distiller)
        elif distiller in ['AlignSoftTarget']:
            self.logical_loss = AlignSoftTargetLoss(hyp, model)
    
    def forward(self, s_p, t_p):
        assert len(s_p) == len(t_p)
        loss = self.logical_loss(s_p, t_p)
        return loss

# ------------------------------BiliBili 魔鬼面具up主------------------------------
class OutputLoss(nn.Module):
    def __init__(self, hyp, distiller='l2'):
        super().__init__()
        
        if distiller == 'l2':
            box_loss = torch.nn.MSELoss(reduction='none')
            cls_loss = torch.nn.MSELoss(reduction='none')
            obj_loss = torch.nn.MSELoss(reduction='none')
        elif distiller == 'l1':
            box_loss = torch.nn.L1Loss(reduction='none')
            cls_loss = torch.nn.L1Loss(reduction='none')
            obj_loss = torch.nn.L1Loss(reduction='none')
        else:
            raise NotImplementedError
        
        self.box_loss = box_loss
        self.cls_loss = cls_loss
        self.obj_loss = obj_loss
        self.hyp = hyp
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def forward(self, s_p, t_p):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        
        for i in range(len(s_p)):
            s_pi, t_pi = s_p[i], t_p[i]
            b, a, gj, gi, _ = s_pi.size()
            
            s_pxywh, s_pobj, s_pcls = s_pi.tensor_split((4, 5), dim=-1)
            t_pxywh, t_pobj, t_pcls = t_pi.tensor_split((4, 5), dim=-1)
            cls_num = s_pcls.size(-1)
            
            t_obj_scale = t_pobj.sigmoid()
            b_obj_scale = t_obj_scale.repeat(1, 1, 1, 1, 4)
            lbox += torch.mean(self.box_loss(s_pxywh, t_pxywh) * b_obj_scale)
            if cls_num > 1:
                c_obj_scale = t_obj_scale.repeat(1, 1, 1, 1, cls_num)
                lcls += torch.mean(self.cls_loss(s_pcls, t_pcls) * c_obj_scale)
            
            lobj += torch.mean(self.obj_loss(s_pobj, t_pobj) * t_obj_scale)
        
        lbox *= self.hyp['box']
        lcls *= self.hyp['cls']
        lobj *= self.hyp['obj']
        
        return (lbox + lcls + lobj) * b

class AlignSoftTargetLoss(nn.Module):
    # BiliBili 魔鬼面具Up
    def __init__(self, hyp, model):
        super().__init__()
        
        self.hyp = hyp
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.na = model.model[-1].na
        self.anchors = model.model[-1].anchors
        self.stride = model.model[-1].stride
    
    def novel_kd_loss(self,
                  pred,
                  soft_label,
                  detach_target=True,
                  beta=1.0):
        # code from https://github.com/TinyTigerPan/BCKD
        r"""Loss function for knowledge distilling using KL divergence.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            T (int): Temperature for distillation.
            detach_target (bool): Remove soft_label from automatic differentiation

        Returns:
            torch.Tensor: Loss tensor with shape (N,).
        """
        assert pred.size() == soft_label.size()
        target = soft_label.sigmoid()
        score = pred.sigmoid()

        if detach_target:
            target = target.detach()

        scale_factor = target - score
        kd_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * scale_factor.abs().pow(beta)
        return kd_loss
    
    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
    
    def forward(self, s_p, t_p):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        
        for i in range(len(s_p)):
            s_pi, t_pi = s_p[i], t_p[i]
            b, a, gj, gi, _ = s_pi.size()
            grid, anchor_grid = self._make_grid(gi, gj, i)
            
            s_pxy, s_pwh, s_pobj, s_pcls = s_pi.tensor_split((2, 4, 5), dim=-1)
            t_pxy, t_pwh, t_pobj, t_pcls = t_pi.tensor_split((2, 4, 5), dim=-1)
            cls_num = s_pcls.size(-1)
            
            # Regression
            s_pxy = s_pxy.sigmoid() * 2 - 0.5
            s_pwh = (s_pwh.sigmoid() * 2) ** 2 * anchor_grid
            s_pbox = torch.cat((s_pxy, s_pwh), -1)
            
            t_pxy = t_pxy.sigmoid() * 2 - 0.5
            t_pwh = (t_pwh.sigmoid() * 2) ** 2 * anchor_grid
            t_pbox = torch.cat((t_pxy, t_pwh), -1)
            
            iou = bbox_iou(s_pbox, t_pbox, CIoU=True).squeeze()  # iou(prediction, target)
            lbox += (1.0 - iou).mean()  # iou loss
            
            lobj += torch.mean(self.novel_kd_loss(s_pobj, t_pobj))
            if cls_num > 1:
                lcls += torch.mean(self.novel_kd_loss(s_pcls, t_pcls))
        
        lbox *= self.hyp['box']
        lcls *= self.hyp['cls']
        lobj *= self.hyp['obj']
        
        return (lbox + lcls + lobj) * b

# ------------------------------BiliBili 魔鬼面具up主------------------------------ 
class FeatureLoss(nn.Module):
    def __init__(self,
                 channels_s,
                 channels_t,
                 distiller='cwd'):
        super(FeatureLoss, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.align_module = nn.ModuleList([
            nn.Conv2d(channel, tea_channel, kernel_size=1, stride=1,
                      padding=0).to(device) if channel != tea_channel else nn.Identity()
            for channel, tea_channel in zip(channels_s, channels_t)
        ])
        self.norm = [
            nn.BatchNorm2d(tea_channel, affine=False).to(device)
            for tea_channel in channels_t
        ]

        if (distiller == 'mimic'):
            self.feature_loss = MimicLoss(channels_s, channels_t)
        elif (distiller == 'mgd'):
            self.feature_loss = MGDLoss(channels_s, channels_t)
        elif (distiller == 'cwd'):
            self.feature_loss = CWDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align_module[idx](s)
            s = self.norm[idx](s)
            t = self.norm[idx](t)
            tea_feats.append(t)
            stu_feats.append(s)

        loss = self.feature_loss(stu_feats, tea_feats)
        return loss

# ------------------------------BiliBili 魔鬼面具up主------------------------------
class MimicLoss(nn.Module):
    def __init__(self, channels_s, channels_t):
        super(MimicLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mse = nn.MSELoss()

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.mse(s, t))
        loss = sum(losses)
        return loss

# ------------------------------BiliBili 魔鬼面具up主------------------------------
class MGDLoss(nn.Module):
    def __init__(self,
                 channels_s,
                 channels_t,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3,
                          padding=1)).to(device) for channel in channels_t
        ])

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss

# ------------------------------BiliBili 魔鬼面具up主------------------------------
class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """
    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape
            # normalize in channel diemension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau,
                                       dim=1)  # [N*C, H*W]

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (
                    self.tau**2)
            # cost = torch.sum(-softmax_pred_T * logsoftmax(s.view(-1, W * H)/self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)

        return loss

