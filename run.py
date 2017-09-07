import cv2
import numpy as np
from core.iPatch import PatchFactory as PF
from utils.data_utils import DataFactory as DF


if __name__ == "__main__":
    patch_engine = PF(data_engine=DF(dataset='CIFAR100'))
    
    image = cv2.imread('_IGP0680 1.jpg', 1)
#    image = cv2.resize(image, (image.shape[1]/2, image.shape[0]/2))
    patch_engine.recreate_image(image)