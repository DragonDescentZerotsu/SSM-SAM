import os
import shutil
import glob
import numpy as np

Data_father_dir = '/root/autodl-tmp/Data/LARGE_SET_no_rot'
all_task_list = os.listdir(Data_father_dir)  # 所有的任务都列在这里
test_task = 'right_kidney'
exclude_task_list = ['USF_shadow', 'SBU_shadow', 'ISTD_shadow', 'polyp', test_task, 'COD10K', 'cup', 'BraTS20',
                         'ISBI', 'transparent', 'all_train_meta_data', 'all_pre_train_data', '__MACOSX', 'large_test', '.DS_Store']
all_train_task_list = list(set(all_task_list) - set(exclude_task_list))

for dirs in ['train', 'eval']:
    for stores in ['img', 'mask']:
        if not os.path.exists(os.path.join(Data_father_dir, 'all_train_meta_data', dirs, stores)):
            os.makedirs(os.path.join(Data_father_dir, 'all_train_meta_data', dirs, stores))
            print(os.path.join(Data_father_dir, 'all_train_meta_data', dirs, stores) + ' is made')
        else:
            files = glob.glob(os.path.join(Data_father_dir, 'all_train_meta_data', dirs, stores) + '/*')
            for file in files:
                if os.path.isfile(file):
                    os.remove(file)
            print(os.path.join(Data_father_dir, 'all_train_meta_data', dirs, stores) + ' cleaned')

for task in all_train_task_list:
    img_list = np.array(os.listdir(os.path.join(Data_father_dir, task, 'train', 'img')))
    mask_list = np.array(os.listdir(os.path.join(Data_father_dir, task, 'train', 'mask')))
    permutation = np.random.permutation(len(img_list))
    img_list = img_list[permutation]
    mask_list = mask_list[permutation]
    for (i, name_img), name_mask in zip(enumerate(img_list), mask_list):
        if i < 50:
            print(i+1, end=' ')
            shutil.copy(os.path.join(Data_father_dir, task, 'train', 'img', name_img),
                        os.path.join(Data_father_dir, 'all_train_meta_data', 'train', 'img', task + '_' + name_img))
            shutil.copy(os.path.join(Data_father_dir, task, 'train', 'mask', name_mask),
                        os.path.join(Data_father_dir, 'all_train_meta_data', 'train', 'mask', task + '_' + name_mask))
        else:
            break
    print(task + ' done')
