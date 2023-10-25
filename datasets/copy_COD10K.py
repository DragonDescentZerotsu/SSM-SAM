import os
import shutil


def main():
    father_dir_path = r'F:\AI_projects\Data_temp\COD10K-v3'
    des_dir_path = r'F:\AI_projects\Data\COD10K'
    for i, name in enumerate(os.listdir(os.path.join(father_dir_path, 'Train', 'Image'))):
        if i < 500:
            shutil.copy(os.path.join(father_dir_path, 'Train', 'Image', name),
                        os.path.join(des_dir_path, 'train', 'img', name))
            shutil.copy(os.path.join(father_dir_path, 'Train', 'GT_Object', name.split('.')[0]+'.png'),
                        os.path.join(des_dir_path, 'train', 'mask', name.split('.')[0]+'.png'))
    for i, name in enumerate(os.listdir(os.path.join(father_dir_path, 'Test', 'Image'))):
        if i < 200:
            shutil.copy(os.path.join(father_dir_path, 'Test', 'Image', name),
                        os.path.join(des_dir_path, 'eval', 'img', name))
            shutil.copy(os.path.join(father_dir_path, 'Test', 'GT_Object', name.split('.')[0]+'.png'),
                        os.path.join(des_dir_path, 'eval', 'mask', name.split('.')[0]+'.png'))


if __name__ == '__main__':
    main()
