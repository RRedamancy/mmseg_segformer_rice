import os
import cv2
import numpy as np

path = 'D:/code/mmsegmentation-0.x/rice-seg-voc/JPEGImages'

def compute(path):
    file_names = os.listdir(path)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    for file_name in file_names:
        print("ing...")
        img = cv2.imread(os.path.join(path, file_name), 1)
        per_image_Bmean.append(np.mean(img[:, :, 0]))
        per_image_Gmean.append(np.mean(img[:, :, 1]))
        per_image_Rmean.append(np.mean(img[:, :, 2]))
    R_mean = np.mean(per_image_Rmean)/255
    G_mean = np.mean(per_image_Gmean)/255
    B_mean = np.mean(per_image_Bmean)/255
    stdR = np.std(per_image_Rmean)/255
    stdG = np.std(per_image_Gmean)/255
    stdB = np.std(per_image_Bmean)/255
    return R_mean, G_mean, B_mean, stdR, stdG, stdB

if __name__ == '__main__':
    stdR, stdG, stdB, B_mean, R_mean, Gmean = compute(path)
    print("R_mean= ", R_mean, "G_mean= ", Gmean, "B_mean=", B_mean, "stdR = ", stdR, "stdG = ", stdG, "stdB =", stdB)
