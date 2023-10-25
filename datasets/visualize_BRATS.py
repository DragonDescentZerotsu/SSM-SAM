import shutil

import numpy as np
import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import imageio  # 转换成图像
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from PIL import Image

'''flair/t1/t1ce/t2相当于不同的模态，seg是分割'''


def main():
    np.set_printoptions(threshold=np.inf)
    father_dir_path = r'F:\AI_projects\Data_temp\BRATS2020\MICCAI_BraTS2020_TrainingData'
    destination_dir = r'F:\AI_projects\Data\BraTS20'
    dirnames = os.listdir(father_dir_path)
    bar = tqdm(dirnames, leave=True, desc='dirs')
    for dirname in bar:
        num = dirname.split('_')[-1]
        if int(num) <= 91:
            continue
        imgs = []
        flair_path = os.path.join(father_dir_path, rf'BraTS20_Training_{num}\BraTS20_Training_{num}_flair.nii.gz')
        t1_path = os.path.join(father_dir_path, rf'BraTS20_Training_{num}\BraTS20_Training_{num}_t1.nii.gz')
        t2_path = os.path.join(father_dir_path, rf'BraTS20_Training_{num}\BraTS20_Training_{num}_t2.nii.gz')
        mask_path = os.path.join(father_dir_path, rf'BraTS20_Training_{num}\BraTS20_Training_{num}_seg.nii.gz')
        for path in [flair_path, t1_path, t2_path, mask_path]:
            imgs.append(np.swapaxes(np.array(nib.load(path).get_fdata()), 0, 2))

        for i, triple in enumerate(zip(imgs[0], imgs[1], imgs[2])):
            if 0.25 * imgs[0].shape[0] < i < 0.75 * imgs[0].shape[0] and i % 5 == 0:
                # plt.imshow(triple[0])
                # plt.show()
                # plt.imshow(triple[1])
                # plt.show()
                # plt.imshow(triple[2])
                # plt.show()
                mask = np.where(imgs[-1][i] > 0, 1, 0)

                # plt.imshow(mask)
                # plt.show()
                if np.where(mask > 0)[0].shape[0] / (mask.shape[0] * mask.shape[1]) < 0.019:
                    continue
                plt.imshow(mask, cmap='gray')
                plt.axis('off')
                # plt.show()
                plt.savefig(os.path.join(destination_dir, 'train', 'mask', f'{num}_{i}.png'), bbox_inches='tight',
                            pad_inches=0)
                triple_img = np.stack(triple, axis=0)
                triple_img = np.swapaxes(np.swapaxes(triple_img, 0, 2), 0, 1) / np.max(triple_img) * 255
                triple_img = Image.fromarray(triple_img.astype('uint8'))

                triple_img.save(os.path.join(destination_dir, 'train', 'img', f'{num}_{i}.jpg'))


def move_to_eval():
    father_dir_path = r'F:\AI_projects\Data\BraTS20'
    initial_img_path = os.path.join(father_dir_path, 'train', 'img')
    img_names = os.listdir(initial_img_path)
    selected_names = random.sample(img_names, 200)
    for name in selected_names:
        shutil.move(os.path.join(father_dir_path, 'train', 'img', name), os.path.join(father_dir_path, 'eval', 'img', name))
        shutil.move(os.path.join(father_dir_path, 'train', 'mask', name.split('.')[0]+'.png'),
                    os.path.join(father_dir_path, 'eval', 'mask', name))



if __name__ == '__main__':
    move_to_eval()
