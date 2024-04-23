import torch
import copy
import torch.nn as nn
import torch.nn.functional as tnf
from ImgEnhance.Nets.mobilenet_v1 import mobilenet_v1

# from efficientnet_pytorch import EfficientNet
from ImgEnhance.Nets.resnet import resnet18, resnet34, resnet50, resnet101


class OutputLayer(nn.Module):
    def __init__(self, fc, num_extra):
        super(OutputLayer, self).__init__()
        self.regular_outputs_layer = fc
        self.num_extra = num_extra
        if num_extra > 0:
            self.extra_outputs_layer = nn.Linear(fc.in_features, num_extra)

    def forward(self, x):
        regular_outputs = self.regular_outputs_layer(x)
        if self.num_extra > 0:
            extra_outputs = self.extra_outputs_layer(x)
        else:
            extra_outputs = None

        return regular_outputs, extra_outputs

class HorizonRegression(nn.Module):
    def __init__(self,
                 num_outputs,
                 backbone,
                 pretrained,
                 curriculum_steps=None,
                 extra_outputs=0,
                 pred_category=False):
        super(HorizonRegression, self).__init__()
        if 'efficientnet' in backbone:
            if pretrained:
                self.model = EfficientNet.from_pretrained(backbone, num_classes=num_outputs)
            else:
                self.model = EfficientNet.from_name(backbone, override_params={'num_classes': num_outputs})
            self.model._fc = OutputLayer(self.model._fc, extra_outputs)
        elif backbone == 'resnet18':
            self.model = resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        elif backbone == 'resnet34':
            self.model = resnet34(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        elif backbone == 'resnet50':
            self.model = resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        elif backbone == 'resnet101':
            self.model = resnet101(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        elif backbone == 'mobilenetv1':
            self.model = mobilenet_v1(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        else:
            raise NotImplementedError()

        self.curriculum_steps = [0, 0, 0, 0] if curriculum_steps is None else curriculum_steps
        self.extra_outputs = extra_outputs
        self.pred_category = pred_category
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, epoch=None, **kwargs):
        output = self.model(x, **kwargs)
        for i in range(len(self.curriculum_steps)):
            if epoch is not None and epoch < self.curriculum_steps[i]:
                output[:, -len(self.curriculum_steps) + i] = 0
        return output

    def decode(self, all_outputs):
        outputs, extra_outputs = all_outputs
        outputs = outputs.reshape(len(outputs), -1, 3)

        outputs[:, :, 0] = self.sigmoid(outputs[:, :, 0])

        return outputs, extra_outputs

    def loss(self,
             outputs,
             target,
             conf_weight=1,
             left_point_weight=1,
             right_point_weight=1,
             poly_weight=1):
        pred, extra_outputs = outputs
        bce = nn.BCELoss()
        mse = nn.MSELoss()
        s = nn.Sigmoid()
        pred = pred.reshape(-1, target.shape[1], 3)

        target_categories, pred_confs = target[:, :, 0].reshape((-1, 1)), s(pred[:, :, 0]).reshape((-1, 1))
        target_right_points, pred_right_points = target[:, :, 2].reshape((-1, 1)), pred[:, :, 2].reshape((-1, 1))
        # target_points, pred_kb = target[:, :, 3:].reshape((-1, target.shape[2] - 3)), pred[:, :, 3:].reshape((-1, pred.shape[2] - 3))
        target_left_points, pred_left_points = target[:, :, 1], pred[:, :, 1]
        # pred_k = torch.reshape(pred_kb[:, 0], (-1, 1))
        # pred_b = torch.reshape(pred_kb[:, 1], (-1, 1))
        target_left_points = target_left_points.reshape((-1, 1))
        pred_left_points = pred_left_points.reshape((-1, 1))

        target_confs = (target_categories > 0).float()
        valid_lanes_idx = target_confs == 1
        # valid_lanes_idx_flat = valid_lanes_idx.reshape(-1)

        left_points_loss = mse(target_left_points, pred_left_points)

        right_points_loss = mse(target_right_points, pred_right_points)

        # traget_xs = target_points[valid_lanes_idx_flat, :target_points.shape[1] // 2]
        # target_ys = target_points[valid_lanes_idx_flat, target_points.shape[1] // 2:]
        # pred_ys = copy.deepcopy(target_ys)
        # valid_ys = target_ys >= 0
        # for i in range(0, traget_xs.shape[0]):
        #     pred_ys[i, :] = pred_k[i] * traget_xs[i, :] + 100*pred_b[i]


        # poly_loss = mse(target_ys[valid_ys], pred_ys[valid_ys])


        # poly_loss = poly_loss * poly_weight
        left_points_loss = left_points_loss * left_point_weight
        right_points_loss = right_points_loss * right_point_weight
        conf_loss = bce(pred_confs, target_confs) * conf_weight

        loss = conf_loss + left_points_loss + right_points_loss  # + poly_loss

        return loss, {
            'conf': conf_loss,
            'left_points_loss': left_points_loss,
            'right_points_loss': right_points_loss
        }
