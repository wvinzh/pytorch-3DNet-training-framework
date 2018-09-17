import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import pandas
import glob


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(label_name_path):
    class_to_idx={}
    label_names = pandas.read_csv(label_name_path,sep='\t')

    for index,row in label_names.iterrows():
        class_to_idx[row['Tag Name']] = row['Tag ID']
    return class_to_idx



def get_video_names_and_annotations(annotation_path):
    video_names = []
    annotations = []
    with open(annotation_path,'r') as f:
        line = f.readline()
        while line:
            # print(line)
            video_names.append(line.strip().split(' ',1)[0])
            label_list = (line.strip().split(' ',1)[1]).split(',')
            annotations.append([int(k) for k in label_list])
            line=f.readline()
    return video_names, annotations


def make_dataset(root_path, annotation_path, label_name_path,subset, sample_duration, sample_step):
    # label_name_path readme.txt （label，name）
    # annotation_path annotation.txt (video_name, label)
    video_names, annotations = get_video_names_and_annotations(annotation_path)
    # class_to_idx: eg:'cat'=1
    class_to_idx = get_class_labels(label_name_path)
    idx_to_class = {}
    # idx_to_class: eg:1='cat'
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    # extract some frames from each video
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))
        #video_names are *.mp4 like
        video_path = os.path.join(root_path, video_names[i].split('.')[0])
        # print(video_path)
        if not os.path.exists(video_path):
            continue

        #total_frames represent how many frames has been extraced in a video
        str_path = os.path.join(video_path, '*.jpg')
        total_frames = len(glob.glob(str_path))
        # print(total_frames)
        if total_frames <= 0:
            continue

        begin_t = 1
        end_t = total_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': total_frames,
            'video_id': video_names[i],
            'label':annotations[i]
        }

        if total_frames <= sample_duration:
            #sample['frame_indices'] = list(range(1, total_frames + 1))
            pass
            #dataset.append(sample)
        else:
            for j in range(1, total_frames, sample_step):
                sample_j = copy.deepcopy(sample)
                if j+sample_duration > total_frames + 1:
                    break
                sample_j['frame_indices'] = list(range(j, j + sample_duration))
                dataset.append(sample_j)

    return dataset, idx_to_class


class ShortVideo(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 label_name_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 sample_step=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path,annotation_path,label_name_path,subset,sample_duration,sample_step)
        
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        video_id = self.data[index]['video_id']
        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)


        target = self.data[index]
        if self.target_transform is not None:
            # print(target)
            target = self.target_transform(target)
        # print(clip.size())
        return clip, target, video_id

    def __len__(self):
        return len(self.data)
