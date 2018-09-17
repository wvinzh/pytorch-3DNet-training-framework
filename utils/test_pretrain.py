import torch

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

get_video_names_and_annotations('/home/zengh/Dataset/AiChallengerFrames/300_val.txt')
