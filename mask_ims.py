curr_dir = '/user_data/vayzenbe/GitHub_Repos/segment-anything'
import sys
import os
sys.path.append("..")
sys.path.append(curr_dir)
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time
from glob import glob as glob
import tqdm
from skimage import io
from scipy.ndimage import binary_fill_holes

stim_dir = sys.argv[1]
target_dir = sys.argv[2]

#example usage:
#python mask_ims.py /user_data/vayzenbe/image_sets/ecoset/val/0001_man /lab_data/behrmannlab/image_sets/masked-ecoset/val/0001_man

#if target dir does not exist, create it
if not os.path.exists(target_dir):
    os.makedirs(target_dir, exist_ok=True)

stim_files = glob(f"{stim_dir}/*.jpg") + glob(f"{stim_dir}/*.JPEG")

#specify center of the image as input point
center = input_point = np.array([[112,112]])
input_label = np.array([1])

#load segmentation model 
sam_checkpoint = f"{curr_dir}/checkpoint/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device='cuda')

predictor = SamPredictor(sam)
print("Model loaded")
print('Starting segmentation...')


start = time.time()
#use tqdm to show progress bar


for im_file in stim_files:
    image = cv2.imread(im_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #resize image to 224x224
    image = cv2.resize(image, (224, 224))

    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    #convert boolean mask to uint8
    masks = masks.astype(np.uint8)
    #sum masks along the channel axis
    mask = np.sum(masks, axis=0)
    #binarize
    mask = (mask > 0).astype(np.uint8)

    #fill holes
    mask = binary_fill_holes(mask).astype(np.uint8)


    im_name = os.path.basename(im_file)
    #convert to 0-255 range
    mask_im =  np.uint8(np.interp(mask, (mask.min(), mask.max()), (0, 255)))


    #save image
    io.imsave(f'{target_dir}/{im_name}', mask_im)
    
    


end = time.time()
print(end - start)