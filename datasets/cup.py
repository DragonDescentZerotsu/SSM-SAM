import os
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import numpy as np

father_dir_path = r'F:\AI_projects\Data_temp\drishti\drishti'
des_dir_path = r'F:\AI_projects\Data\cup'

for i, name in enumerate(os.listdir(os.path.join(father_dir_path, 'images'))):
    mask = Image.open(os.path.join(father_dir_path, 'masks', name))
    mask = np.array(mask)[..., 1]
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    if i < 75:
        Image.fromarray(mask).save(os.path.join(des_dir_path, 'train', 'mask', name))
        shutil.copy(os.path.join(father_dir_path, 'images', name), os.path.join(des_dir_path, 'train', 'img', name))
    else:
        Image.fromarray(mask).save(os.path.join(des_dir_path, 'eval', 'mask', name))
        shutil.copy(os.path.join(father_dir_path, 'images', name), os.path.join(des_dir_path, 'eval', 'img', name))

