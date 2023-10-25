import os
import shutil

father_dir_path = r'F:\AI_projects\Data_temp\polyp'
des_dir_path = r'F:\AI_projects\Data\polyp'

for i, name in enumerate(os.listdir(os.path.join(father_dir_path, 'images'))):
    if i<500:
        shutil.copy(os.path.join(father_dir_path, 'images', name), os.path.join(des_dir_path, 'train', 'img', name))
        shutil.copy(os.path.join(father_dir_path, 'masks', name), os.path.join(des_dir_path, 'train', 'mask', name))
    elif i<700:
        shutil.copy(os.path.join(father_dir_path, 'images', name), os.path.join(des_dir_path, 'eval', 'img', name))
        shutil.copy(os.path.join(father_dir_path, 'masks', name), os.path.join(des_dir_path, 'eval', 'mask', name))
