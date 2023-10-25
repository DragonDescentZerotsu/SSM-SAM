import os
from PIL import Image
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt


def get_number(filename):
    base_name = os.path.basename(filename)  # 获取文件名（不包含路径）
    name, ext = os.path.splitext(base_name)  # 分割文件名和扩展名（如 .jpg）
    name = name.split('_')[0]
    return int(name)  # 转换文件名（不包含扩展名）为整数，并返回


def move_main():
    source_father_data_dir = '/Users/kirianozan/Documents/AI_project/Data/MRI_T2_raw'
    des_father_data_dir = '/Users/kirianozan/Documents/AI_project/Data/MRI_T2_new'
    exclude_task_list = ['USF_shadow', 'SBU_shadow', 'ISTD_shadow', 'polyp', 'COD10K', 'cup', 'BraTS20','all_pre_train_data',
                         'ISBI', 'transparent', 'all_train_meta_data', '__MACOSX', 'large_test', '.DS_Store']
    all_task_list = os.listdir(source_father_data_dir)  # 所有的任务都列在这里
    all_task_list = list(set(all_task_list) - set(exclude_task_list))
    batch_size = 4
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
        img_list = os.listdir(os.path.join(organ_source_dir, 'img'))
        if '.DS_Store' in img_list:
            img_list.remove('.DS_Store')
        img_list = sorted(img_list, key=get_number)  # 排序好之后再移动
        # mask_list = sorted(os.listdir(os.path.join(organ_source_dir, 'mask')), key=get_number)

        # print(source_dir)

        chunk_list = []
        middle_slice = None
        alphabeta = ['a', 'b', 'c', 'd', 'e']
        for i, img in enumerate(img_list):
            if i > len(img_list) - chunk_len:
                break
            num = img.split('.')[0]
            if i % chunk_len != 0 or i == 0:
                if len(chunk_list) == 2 and middle_slice is None:
                    middle_slice = num
                else:
                    chunk_list.append(num)
            else:
                for j in range(batch_size):
                    shutil.copy(os.path.join(organ_source_dir, 'img', middle_slice + '.jpg'),
                                os.path.join(organ_des_path, 'support_train', 'img',
                                             middle_slice + '_' + alphabeta[j] + '.jpg'))
                    shutil.copy(os.path.join(organ_source_dir, 'mask', middle_slice + '.png'),
                                os.path.join(organ_des_path, 'support_train', 'mask',
                                             middle_slice + '_' + alphabeta[j] + '.png'))
                for j, img_c in enumerate(chunk_list):
                    shutil.copy(os.path.join(organ_source_dir, 'img', img_c + '.jpg'),
                                os.path.join(organ_des_path, 'query_train', 'img', img_c + '.jpg'))
                    shutil.copy(os.path.join(organ_source_dir, 'mask', img_c + '.png'),
                                os.path.join(organ_des_path, 'query_train', 'mask', img_c + '.png'))
                chunk_list = [num]
                middle_slice = None
        print(organ + ' done')


def move_train_to_eval():
    """前面只有train里面移动了有图片，eval还没有图片，按照一定比例移动过去"""
    eval_img_num = 50
    batch_size = 4
    father_data_dir = '/Users/kirianozan/Documents/AI_project/Data/MRI_T2_new'
    all_task_list = os.listdir(father_data_dir)  # 所有的任务都列在这里
    exclude_task_list = ['USF_shadow', 'SBU_shadow', 'ISTD_shadow', 'polyp', 'COD10K', 'cup', 'BraTS20','all_pre_train_data',
                         'ISBI', 'transparent', 'all_train_meta_data', '__MACOSX', 'large_test', '.DS_Store']
    all_task_list = list(set(all_task_list) - set(exclude_task_list))
    for organ in all_task_list:
        for part in ['support', 'query']:
            i = 0
            img_train_path = os.path.join(father_data_dir, organ, part + '_train', 'img')
            mask_train_path = os.path.join(father_data_dir, organ, part + '_train', 'mask')
            img_eval_path = os.path.join(father_data_dir, organ, part + '_eval', 'img')
            mask_eval_path = os.path.join(father_data_dir, organ, part + '_eval', 'mask')
            img_list = sorted(os.listdir(img_train_path), key=get_number)  # 排序好之后再移动
            mask_list = sorted(os.listdir(mask_train_path), key=get_number)
            while i <= eval_img_num - batch_size:
                for j in range(batch_size):
                    shutil.move(os.path.join(img_train_path, img_list[i + j]),
                                os.path.join(img_eval_path, img_list[i + j]))
                    shutil.move(os.path.join(mask_train_path, mask_list[i + j]),
                                os.path.join(mask_eval_path, mask_list[i + j]))
                i += batch_size
        print(organ + ' done')


def create_gaussian_at_point(center, shape, sigma):
    """Generate a gaussian centered at a specific point in a matrix of a given shape."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    y_center, x_center = center
    g = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma ** 2))
    return g


def blur(mask_path):
    img_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.normalize(img_gray.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

    blurred_mask = cv2.GaussianBlur(mask, (101, 101), 0)
    max_val = np.max(blurred_mask)
    blurred_mask[mask == 1] = max_val
    for i in range(3):
        blurred_mask = cv2.GaussianBlur(blurred_mask, (51, 51), 0)
        max_val = np.max(blurred_mask)
        blurred_mask[mask == 1] = max_val


    blurred_mask = cv2.normalize(blurred_mask, None, 0, 255, cv2.NORM_MINMAX)
    return blurred_mask


def blur_mask():
    """前面把图片都移动到train、eval中了，现在开始高斯模糊mask"""
    father_data_dir = '/Users/kirianozan/Documents/AI_project/Data/MRI_T2_new'
    all_task_list = os.listdir(father_data_dir)  # 所有的任务都列在这里
    exclude_task_list = ['USF_shadow', 'SBU_shadow', 'ISTD_shadow', 'polyp', 'COD10K', 'cup', 'BraTS20', 'all_pre_train_data',
                         'ISBI', 'transparent', 'all_train_meta_data', '__MACOSX', 'large_test', '.DS_Store']
    finished = ['left_kidney', 'aorta', 'LAG', 'RAG', 'gallbladder', 'stomach', 'pancreas', 'esophagus']
    all_task_list = list(set(all_task_list) - set(exclude_task_list))  # - set(finished))
    for organ in all_task_list:
        for part in ['_train', '_eval']:
            # if organ != 'liver':
            #     continue
            # else:
            #     if part != '_train':
            #         continue
            print('\t' + organ + ' blurring' + part)
            mask_path = os.path.join(father_data_dir, organ, 'support' + part, 'mask')
            mask_list = sorted(os.listdir(mask_path), key=get_number)
            old_num = 0
            count = 0
            for mask in mask_list:
                # if count == 200:
                #     break
                if old_num == get_number(mask):
                    blurred_mask.save(os.path.join(mask_path, mask))
                else:
                    old_num = get_number(mask)
                    blurred_mask = blur(os.path.join(mask_path, mask))
                    blurred_mask = blurred_mask.astype(np.uint8)
                    blurred_mask = Image.fromarray(blurred_mask, 'L')
                    blurred_mask.save(os.path.join(mask_path, mask))
                # count += 1
        print('\t\t' + organ + ' blurred')


if __name__ == '__main__':
    # move_main()
    move_train_to_eval()
    print('start blurring')
    blur_mask()

    # blurred_mask = blur('/Users/kirianozan/Documents/AI_project/Data/LARGE_SET_new/liver/query_train/mask/157.png')
    # plt.imshow(blurred_mask, cmap='gray')
    # plt.show()
