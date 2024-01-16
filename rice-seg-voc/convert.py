import os
from PIL import Image
import numpy as np


data_root = "D:/code/mmsegmentation-0.x/rice-seg-voc/SegmentationClass"
save_root = "D:/code/mmsegmentation-0.x/rice-seg-voc/SegmentationClassRaw"

if __name__ == "__main__":
    files = os.listdir(data_root)
    for file in files:
        path = os.path.join(data_root, file)
        print("path: ", path)
        image = Image.open(path).convert('L')
        img = np.array(image)
        img[img != 0] = 254
        # print(np.shape(img))

        save_path = os.path.join(save_root, file)
        image = Image.fromarray(img)
        image.save(save_path)
        print("save_path: ", save_path)