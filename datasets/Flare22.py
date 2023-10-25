import os
import shutil

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
# import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    writer = SummaryWriter(log_dir='tensorboard')
    organ_list = ['liver', 'right_kidney', 'spleen', 'pancreas', 'aorta', 'IVC', 'RAG', 'LAG', 'gallbladder',
                  'esophagus', 'stomach', 'duodenum', 'left_kidney']
    father_dir_path = '/Users/kirianozan/Documents/AI_project/Data_temp/FLARE22Train'
    des_dir_path = '/Users/kirianozan/Documents/AI_project/Data/LARGE_SET_raw'
    img_number_dict = {}
    for organ in organ_list:
        temp_dict = {'train': 0, 'eval': 0}
        img_number_dict[organ] = temp_dict
        if not os.path.exists(os.path.join(des_dir_path, organ)):  # 如果这些路径不存在就先创建
            for dirs in ['train', 'eval']:
                for diff in ['img', 'mask']:
                    os.makedirs(os.path.join(des_dir_path, organ, dirs, diff))

    for file in os.listdir(os.path.join(father_dir_path, 'images')):
        imgs = nib.load(os.path.join(father_dir_path, 'images', file)).get_fdata()
        mask_name = '_'.join(file.split('.')[0].split('_')[:-1]) + '.nii.gz'
        masks = nib.load(os.path.join(father_dir_path, 'labels', mask_name)).get_fdata()
        # masks = torch.from_numpy(masks).to(device)
        for i in range(imgs.shape[-1]):
            # if i % 4 == 0:
            img = imgs[..., i]
            img = (((img - np.min(img)) / (np.max(img) - np.min(img))) * 255).astype(np.uint8)
            mask = masks[..., i]
            # plt.imshow(mask)
            # plt.show()
            for c, organ in enumerate(organ_list):
                for k, v in img_number_dict.items():
                    writer.add_scalar(k + 'train', v['train'], int(i / 4))
                    writer.add_scalar(k + 'eval', v['eval'], int(i / 4))
                    writer.flush()
                if np.where(mask == c + 1)[0].shape[0] / (mask.shape[0] * mask.shape[1]) > 0.0001:
                    mask_c = np.where(mask == c + 1, 255, 0)
                    # turn = random.choice([0, 1, 2, 3])
                    # img_out = np.rot90(img, turn)
                    # mask_c = np.rot90(mask_c, turn)
                    img_out = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                    if img_number_dict[organ]['train'] < 100000:
                        name_num = img_number_dict[organ]['train']
                        Image.fromarray(img_out).save(
                            os.path.join(des_dir_path, organ, 'train', 'img', f'{name_num + 1}.jpg'))
                        Image.fromarray(mask_c.astype(np.uint8)).save(
                            os.path.join(des_dir_path, organ, 'train', 'mask', f'{name_num + 1}.png'))
                        img_number_dict[organ]['train'] += 1
                    elif img_number_dict[organ]['eval'] < 100000:
                        name_num = img_number_dict[organ]['eval']
                        Image.fromarray(img_out).save(
                            os.path.join(des_dir_path, organ, 'eval', 'img', f'{name_num + 1}.jpg'))
                        Image.fromarray(mask_c.astype(np.uint8)).save(
                            os.path.join(des_dir_path, organ, 'eval', 'mask', f'{name_num + 1}.png'))
                        img_number_dict[organ]['eval'] += 1


def move_to_eval():
    dir_path = '/Users/kirianozan/Documents/AI_project/Data/LARGE_SET_no_rot'
    organ_list = ['liver', 'right_kidney', 'spleen', 'pancreas', 'aorta', 'IVC', 'RAG', 'LAG', 'gallbladder',
                  'esophagus', 'stomach', 'duodenum', 'left_kidney']
    for organ in organ_list:
        img_list = os.listdir(os.path.join(dir_path, organ, 'train', 'img'))
        for i, img in enumerate(img_list):
            num = img.split('.')[0]
            mask = num + '.png'
            if i < 125:
                shutil.move(os.path.join(dir_path, organ, 'train', 'img', img), os.path.join(dir_path, organ, 'eval', 'img', img))
                shutil.move(os.path.join(dir_path, organ, 'train', 'mask', mask), os.path.join(dir_path, organ, 'eval', 'mask', mask))
            else:
                break


if __name__ == '__main__':
    main()
    # move_to_eval()
