from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def red():
    des_dir_path = r'F:\AI_projects\Data\transparent'
    source_dir_path = r'F:\AI_projects\Data_temp\transparent'
    for dir in os.listdir(source_dir_path):
        for i, name in enumerate(os.listdir(os.path.join(source_dir_path, dir, 'images'))):
            # if dir=='easy' or (i < 110 and dir == 'hard'):
            #      continue
            mask = Image.open(os.path.join(source_dir_path, dir, 'masks', name.split('.')[0] + '_mask.png'))
            mask = torch.from_numpy(np.array(mask)).to(device)
            # mask = np.where(np.array(mask)[..., 0] > 0, 255, 0)
            mask = torch.where(mask[..., 0] > 0, 255, 0)
            # plt.imshow(mask, cmap='gray')
            # plt.axis('off')
            if (i < 300 and dir == 'easy') or (i<200 and dir == 'hard'):
                shutil.copy(os.path.join(source_dir_path, dir, 'images', name),
                            os.path.join(des_dir_path, 'train', 'img', name))

                # plt.savefig(os.path.join(des_dir_path, 'train', 'mask', name.split('.')[0] + '.png'),
                #             bbox_inches='tight',
                #             pad_inches=0)
                Image.fromarray(mask.cpu().numpy().astype(np.uint8)).save(os.path.join(des_dir_path, 'train', 'mask', name.split('.')[0] + '.png'))
                # shutil.copy(os.path.join(source_dir_path, dir, 'masks', name.split('.')[0] + '_mask.png'),
                #             os.path.join(des_dir_path, 'train', 'img', name.split('.')[0] + '.png'))
            elif i < 500 and dir == 'easy':
                shutil.copy(os.path.join(source_dir_path, dir, 'images', name),
                            os.path.join(des_dir_path, 'eval', 'img', name))
                # plt.savefig(os.path.join(des_dir_path, 'eval', 'mask', name.split('.')[0] + '.png'),
                #             bbox_inches='tight',
                #             pad_inches=0)
                Image.fromarray(mask.cpu().numpy().astype(np.uint8)).save(os.path.join(des_dir_path, 'eval', 'mask', name.split('.')[0] + '.png'))
                # shutil.copy(os.path.join(source_dir_path, dir, 'masks', name.split('.')[0] + '_mask.png'),
                #             os.path.join(des_dir_path, 'eval', 'img', name.split('.')[0] + '.png'))
            else:
                break

    # img = Image.open(r'F:\AI_projects\Data_temp\transparent\easy\masks\148_mask.png')
    # img = np.array(img)
    # img1 = img[..., 3]
    # img2 = np.where(img1 == 255, 0, 1)
    # plt.imshow(img2)
    # plt.show()
    # print(img)


# def white():

if __name__ == '__main__':
    red()
