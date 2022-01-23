from json import load
import sys
import time
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from Data import load_img
from Training_functions import pixel_accuracy, mIoU
from Simple_Unet import *

data = None
with open(sys.argv[1],'rb') as file:
    data = file.read()
img = load_img(data)
mask = to_tensor(np.load(sys.argv[2]))
model = UNet(num_class=7)
model.load_state_dict(torch.load('dict.pth'))
model.eval()
start = time.perf_counter()
res = model(img)
end = time.perf_counter()
acc = pixel_accuracy(res,mask)
iou = mIoU(res,mask,7)
print('Accuracy: {}, IoU: {}, time elapsed: {}'.format(acc,iou,end-start))
