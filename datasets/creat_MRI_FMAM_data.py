import os
import shutil


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


def move_img():
    source_father_data_dir = '/Users/kirianozan/Documents/AI_project/Data/MRI_T2_raw'
    des_father_data_dir = '/Users/kirianozan/Documents/AI_project/Data/MRI_T2_new'

    exclude_task_list = ['USF_shadow', 'SBU_shadow', 'ISTD_shadow', 'polyp', 'COD10K', 'cup', 'BraTS20',
                         'all_pre_train_data',
                         'ISBI', 'transparent', 'all_train_meta_data', '__MACOSX', 'large_test', '.DS_Store']
    all_task_list = os.listdir(source_father_data_dir)  # 所有的任务都列在这里
    all_task_list = list(set(all_task_list) - set(exclude_task_list))
    batch_size = 2
    chunk_len = batch_size + 1

    for organ in all_task_list:
        for part in ['query_train', 'query_eval', 'support_train', 'support_eval']:
            for rule in ['img', 'mask']:
                des_path = os.path.join(des_father_data_dir, organ, part, rule)
                if not os.path.exists(des_path):
                    os.makedirs(des_path)
                    print(des_path + ' created')
                else:
                    for file in os.listdir(des_path):
                        os.unlink(os.path.join(des_path, file))
                    print(des_path + ' cleaned')

        organ_source_dir = os.path.join(source_father_data_dir, organ, 'train')
        organ_des_path = os.path.join(des_father_data_dir, organ)
        print(organ)
        img_list = os.listdir(os.path.join(organ_source_dir, 'img'))  # 某一个器官的所有图片
        mask_list = os.listdir(os.path.join(organ_source_dir, 'mask'))
        if '.DS_Store' in img_list:
            img_list.remove('.DS_Store')
        img_list = sorted(img_list, key=get_number)  # 排序好之后再移动

        i = 1  # 从第二张图片开始
        for support_img in img_list[1:]:
            num_support = support_img.split('.')[0]
            if i == len(img_list) - 1:
                break
            query_img_1 = img_list[i - 1]
            query_img_2 = img_list[i + 1]
            for place in ['a', 'b']:
                shutil.copy(os.path.join(organ_source_dir, 'img', f'{num_support}.jpg'),
                            os.path.join(organ_des_path, 'support_train', 'img', f'{num_support}_{place}.jpg'))
                shutil.copy(os.path.join(organ_source_dir, 'mask', f'{num_support}.png'),
                            os.path.join(organ_des_path, 'support_train', 'mask', f'{num_support}_{place}.png'))
            for query_img in [query_img_1, query_img_2]:
                num = query_img.split('.')[0]
                shutil.copy(os.path.join(organ_source_dir, 'img', f'{num}.jpg'),
                            os.path.join(organ_des_path, 'query_train', 'img', f'{num_support}_{num}.jpg'))
                shutil.copy(os.path.join(organ_source_dir, 'mask', f'{num}.png'),
                            os.path.join(organ_des_path, 'query_train', 'mask', f'{num_support}_{num}.png'))
            i += 1


if __name__ == '__main__':
    move_img()
