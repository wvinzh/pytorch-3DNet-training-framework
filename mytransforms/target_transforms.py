import random
import math

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        dst = []
        for t in self.transforms:
            dst.append(t(target))
        return dst


class ClassLabel(object):

    def __call__(self, target):
        return target['label']

class MultiLabelTransForm(object):
    def __init__(self,class_num=10):
        self.class_num = class_num

    def __call__(self, target):
        import torch
        one_hot_label =  torch.zeros(self.class_num)
        one_hot_label[target['label']]=1
        return one_hot_label

class VideoID(object):

    def __call__(self, target):
        return target['video_id']
