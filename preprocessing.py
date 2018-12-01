from __future__ import division, print_function
import numpy as np
np.random.seed(98764)
from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
import os
from PIL import Image

image_dir = './images'
ground_truth_dir = './gts'
image_file_list = []
i = 0
for subdir, dirs, files in os.walk(image_dir):
    for file in files:
        if(file.split('.')[1] == 'tif'):
            print(os.path.join(subdir, file))
            image_file_list.append(os.path.join(subdir, file))
            i= i+1
print('total number of .tif files is '+ str(i))
for file in image_file_list:
    folder_name = "."+file.split(".")[1]
    print(folder_name)
    folder_name = folder_name.replace("images","gts")
    print(folder_name)
    for subdir, dirs, files in os.walk(folder_name):
        cur_gts = []
        for file in files:
            gt = Image.open(os.path.join(subdir, file))
            imarray = np.array(gt)
            imarray[imarray>=1] = 1
            cur_gts.append(imarray)
        consensus_gt = reduce((lambda x, y: x + y), cur_gts)
        print(np.max(consensus_gt))
        consensus_gt[consensus_gt<2] = 0
        consensus_gt[consensus_gt>=2] = 1
        print(np.max(consensus_gt))
        im_save = Image.fromarray(consensus_gt)
        result_path = subdir.replace('gts','processed_gts')
        result_folder_path = reduce((lambda x, y: x + '/'+y), result_path.split("/")[:-1])
        try:
            print(result_folder_path)
            os.mkdir(result_folder_path)
        except:
            print(result_folder_path+'already exist')
        im_save.save(result_path+'.tif')
        print(result_path+'.tif'+' is saved')
