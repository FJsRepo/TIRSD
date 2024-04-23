import cv2
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
from imgaug.augmentables.lines import LineString, LineStringsOnImage

from .HorizonSet import HorizonNet

# HorizonSet_MEAN = np.array([-0.0138, -0.0138, -0.0138])
# HorizonSet_STD = np.array([0.9587, 0.9587, 0.9587])

HorizonSet_MEAN = np.array([0.491, 0.491, 0.491])
HorizonSet_STD = np.array([0.228, 0.228, 0.228])

red = (0, 0, 255) # gt color
blue = (255, 0, 0)
green = (0, 255, 0) # pred color

class HorizonDataset(Dataset):
    def __init__(self,
                 dataset='HorizonSet',
                 augmentations=None,
                 normalize=False,
                 split='train',
                 img_size=(512, 640),
                 aug_chance=1.,
                 **kwargs):
        super(HorizonDataset, self).__init__()
        if dataset == 'HorizonSet':
            self.dataset = HorizonNet(split=split, **kwargs)
            print("----------------------------")
            print("self.dataset:", self.dataset)
            print("----------------------------")
        else:
            raise NotImplementedError()
        self.transform_annotations()
        self.img_h, self.img_w = img_size

        if augmentations is not None:
            # add augmentations
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in augmentations]  # add augmentation

        self.normalize = normalize
        transformations = iaa.Sequential([Resize({'height': self.img_h, 'width': self.img_w})])
        self.to_tensor = ToTensor()
        self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=aug_chance), transformations])
        self.max_Horizon = self.dataset.Horizon_num

    def transform_annotation(self, anno, img_wh=None):
        if img_wh is None:
            img_h = self.dataset.get_img_heigth(anno['path'])
            img_w = self.dataset.get_img_width(anno['path'])
        else:
            img_w, img_h = img_wh
        old_Horizon = anno['Horizon']

        categories = [1]
        old_Horizon = zip(old_Horizon, categories)
        Horizon = np.ones((self.dataset.Horizon_num, 1 + 2 + 2 * self.dataset.max_points), dtype=np.float32) * -1e5
        Horizon[:, 0] = 0
        for Horizon_pos, (Horizon_coordinate, category) in enumerate(old_Horizon):
            lower, upper = Horizon_coordinate[0][1], Horizon_coordinate[-1][1]
            xs = np.array([p[0] for p in Horizon_coordinate]) / img_w
            ys = np.array([p[1] for p in Horizon_coordinate]) / img_h
            Horizon[Horizon_pos, 0] = category
            Horizon[Horizon_pos, 1] = lower / img_h
            Horizon[Horizon_pos, 2] = upper / img_h
            Horizon[Horizon_pos, 3:3 + len(xs)] = xs
            Horizon[Horizon_pos, (3 + self.dataset.max_points):(3 + self.dataset.max_points + len(ys))] = ys
        new_anno = {
            'path': anno['path'],
            'label': Horizon,
            'old_anno': anno,
            'categories': [cat for _, cat in old_Horizon]
        }
        return new_anno

    @property
    def annotations(self):
        return self.dataset.annotations
    def transform_annotations(self):
        print('Transforming annotations...')
        self.dataset.annotations = np.array(list(map(self.transform_annotation, self.dataset.annotations)))
        print('Done.')

    def draw_annotation(self, idx, pred=None, img=None, cls_pred=None):
        if img is None:
            img, label, _ = self.__getitem__(idx, transform=True)
            img = img.permute(1, 2, 0).numpy()
            if self.normalize:
                img = img * np.array(HorizonSet_STD) + np.array(HorizonSet_MEAN)
            img = (img * 255).astype(np.uint8)
            img = np.ascontiguousarray(img)

        else:
            _, label, _ = self.__getitem__(idx)

        img_h, img_w, _ = img.shape

        # Draw GT label
        for i, Horizon_coordinate in enumerate(label):
            Horizon_coordinate = Horizon_coordinate[3:]
            xs = Horizon_coordinate[: len(Horizon_coordinate) // 2]
            ys = Horizon_coordinate[len(Horizon_coordinate) // 2:]
            gt_ys = ys  # * img_h
            gt_xs = xs  # * img_w

            gt_left = (0, round(gt_ys[0] * img_h))
            gt_right = (639, round(gt_ys[-1] * img_h))

        img = cv2.line(img, gt_left, gt_right, green, thickness=1, lineType=cv2.LINE_AA)

        if pred is None:
            return img

        # Draw pred label
        for i, Horizon_coordinate in enumerate(pred):
            Horizon_coordinate = Horizon_coordinate[1:]  # remove conf
            lefty, righty = Horizon_coordinate[0], Horizon_coordinate[1]  # left_endpoint, right_endpoint
            # 基于端点的形式计算对应的直线的kb从而与ys_gt进行偏差计算
            # 基于归一化后数据的计算
            pred_left = (0, lefty)
            pred_right = (1, righty)
            if pred_left[1] == pred_right[1]:
                k = 0
                b = pred_left[1]
            else:
                k = (pred_right[1] - pred_left[1]) / (pred_right[0] - pred_left[0])
                b = pred_left[1]

            pred_left = (0, round(lefty * img_h))
            pred_right = (639, round(righty * img_h))

        img = cv2.line(img, pred_left, pred_right, red, thickness=1, lineType=cv2.LINE_AA)

        # 计算偏差
        totalDiff = []
        for i in range(len(gt_xs)):
            diff = abs(gt_ys[i] - (k * gt_xs[i] + b)) * img_h
            totalDiff.append(diff)

        avgDiffOneImg = np.mean(totalDiff)

        return img, avgDiffOneImg

    def Horizon_to_linestrings(self, Horizon):
        lines = []
        for Horizon_line in Horizon:
            lines.append(LineString(Horizon_line))
        return lines

    def linestrings_to_Horizon(self, lines):
        Horizon = []
        for line in lines:
            Horizon.append(line.coords)
        return Horizon

    def __getitem__(self, idx, transform=True):
        item = self.dataset[idx]
        img = cv2.imread(item['path'])
        label = item['label']
        if transform:
            line_strings = self.Horizon_to_linestrings(item['old_anno']['Horizon'])
            line_strings = LineStringsOnImage(line_strings, shape=img.shape)
            img, line_strings = self.transform(image=img, line_strings=line_strings)
            line_strings.clip_out_of_image_()
            new_anno = {'path': item['path'], 'Horizon': self.linestrings_to_Horizon(line_strings)}
            new_anno['categories'] = item['categories']
            label = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']

        img = img / 255.
        if self.normalize:
            img = (img - HorizonSet_MEAN) / HorizonSet_STD
        img = self.to_tensor(img.astype(np.float32))
        return (img, label, idx)

    def __len__(self):
        return len(self.dataset)

def main():
    import torch
    from lib.config import Config
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = Config('config.yaml')
    train_dataset = cfg.get_dataset('train')
    for idx in range(len(train_dataset)):
        img = train_dataset.draw_annotation(idx)
        cv2.imshow('sample', img)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
