import os
from PIL import Image
import numpy as np


train_data_root = "/data/VOC-rice-seg/SegmentationClass"
val_data_root = "/data/VOC-rice-seg-test/SegmentationClass"
save_split_txt_root = "/data/VOC-rice-seg/ImageSets/Segmentation"

if __name__ == "__main__":
    train_filename_list = [os.path.splitext(filename)[0] for filename in os.listdir(train_data_root)]
    val_filename_list = [os.path.splitext(filename)[0] for filename in os.listdir(val_data_root)]

    with open(os.path.join(save_split_txt_root, 'train.txt'), 'w') as f:
        for filename in train_filename_list:
            f.write(filename + '\n')
    
    with open(os.path.join(save_split_txt_root, 'val.txt'), 'w') as f:
        for filename in val_filename_list:
            f.write(filename + '\n')