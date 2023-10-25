import os
import shutil
import glob
import numpy as np

def get_number(filename):
    base_name = os.path.basename(filename)  # 获取文件名（不包含路径）
    name = base_name.split('.')[0].split('_')[0]  # 分割文件名和扩展名（如 .jpg）
    if not name.isdigit():
        name = base_name.split('.')[0].split('_')[-1]  # 这里是为了适配all_train_meta_data数据名
    try:
        return int(name)  # 转换文件名（不包含扩展名）为整数，并返回
    except ValueError:
        print(base_name)
        return None

def remove_DS_Store(all_list):
    re_list = []
    for file_list in all_list:
        if '.DS_Store' in file_list:
            file_list.remove('.DS_Store')
        re_list.append(sorted(file_list, key=get_number))
    return re_list

Data_father_dir = '/Users/kirianozan/Documents/AI_project/Data/MRI_T2_new'
all_task_list = os.listdir(Data_father_dir)  # 所有的任务都列在这里
test_task = 'spleen'
exclude_task_list = ['USF_shadow', 'SBU_shadow', 'ISTD_shadow', 'polyp', test_task, 'COD10K', 'cup', 'BraTS20',
                         'ISBI', 'transparent', 'all_train_meta_data', 'all_pre_train_data', '__MACOSX', 'large_test', '.DS_Store']
all_train_task_list = list(set(all_task_list) - set(exclude_task_list))

for dirs in ['support_train', 'support_eval', 'query_train', 'query_eval']:
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

j = 1
for task in all_train_task_list:
    img_list = os.listdir(os.path.join(Data_father_dir, task, 'query_train', 'img'))
    mask_list = os.listdir(os.path.join(Data_father_dir, task, 'query_train', 'mask'))
    support_img_list = os.listdir(os.path.join(Data_father_dir, task, 'support_train', 'img'))
    gaussian_mask_list = os.listdir(os.path.join(Data_father_dir, task, 'support_train', 'mask'))
    img_list, mask_list, gaussian_mask_list = remove_DS_Store([img_list, mask_list, gaussian_mask_list])  # 排好序了

    # permutation = np.random.permutation(len(img_list))  # 这里不需要随机排列了
    # img_list = img_list[permutation]
    # mask_list = mask_list[permutation]
    for i, (query_img, query_mask, support_img, gaussian_mask) in enumerate(zip(img_list, mask_list, support_img_list, gaussian_mask_list)):
        if i < 200:
            print(i+1, end=' ')
            shutil.copy(os.path.join(Data_father_dir, task, 'query_train', 'img', query_img),
                        os.path.join(Data_father_dir, 'all_train_meta_data', 'query_train', 'img', str(j)+'.jpg'))
            shutil.copy(os.path.join(Data_father_dir, task, 'query_train', 'mask', query_mask),
                        os.path.join(Data_father_dir, 'all_train_meta_data', 'query_train', 'mask', str(j)+'.png'))
            shutil.copy(os.path.join(Data_father_dir, task, 'support_train', 'img', support_img),
                        os.path.join(Data_father_dir, 'all_train_meta_data', 'support_train', 'img', str(j)+'.jpg'))
            shutil.copy(os.path.join(Data_father_dir, task, 'support_train', 'mask', gaussian_mask),
                        os.path.join(Data_father_dir, 'all_train_meta_data', 'support_train', 'mask', str(j)+'.png'))
        else:
            break
        j += 1
    print(task + ' done')
