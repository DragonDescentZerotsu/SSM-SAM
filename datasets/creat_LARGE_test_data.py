import os
import shutil


def get_number(filename):
    base_name = os.path.basename(filename)  # 获取文件名（不包含路径）
    name, ext = os.path.splitext(base_name)  # 分割文件名和扩展名（如 .jpg）
    return int(name)  # 转换文件名（不包含扩展名）为整数，并返回


test_task = 'left_kidney'
chunk_len = 6
batch_size = 5

father_data_dir = '/root/autodl-tmp/Data/LARGE_SET_no_rot'
for part in ['support', 'query']:
    for rule in ['img', 'mask']:
        des_path = os.path.join(father_data_dir, 'large_test', part, rule)
        if not os.path.exists(des_path):
            os.makedirs(des_path)
            print('created')
        else:
            for file in os.listdir(des_path):
                os.unlink(os.path.join(des_path, file))
            print('cleaned')

source_dir = os.path.join(father_data_dir, test_task, 'train')
des_path = os.path.join(father_data_dir, 'large_test')
img_list = sorted(os.listdir(os.path.join(source_dir, 'img')), key=get_number)  # 排序好之后再移动
mask_list = sorted(os.listdir(os.path.join(source_dir, 'mask')), key=get_number)

print(source_dir)

chunk_list = []
middle_slice = None
alphabeta = ['a', 'b', 'c', 'd', 'e']
for i, img in enumerate(img_list):
    if i > len(img_list) - chunk_len:
        break
    num = img.split('.')[0]
    if i % 6 != 0 or i == 0:
        if len(chunk_list) == 2 and middle_slice is None:
            middle_slice = num
        else:
            chunk_list.append(num)
    else:
        for j in range(batch_size):
            shutil.copy(os.path.join(source_dir, 'img', middle_slice + '.jpg'),
                        os.path.join(des_path, 'support', 'img', middle_slice + '_' + alphabeta[j] + '.jpg'))
            shutil.copy(os.path.join(source_dir, 'mask', middle_slice + '.png'),
                        os.path.join(des_path, 'support', 'mask', middle_slice + '_' + alphabeta[j] + '.png'))
        for j, img_c in enumerate(chunk_list):
            shutil.copy(os.path.join(source_dir, 'img', img_c + '.jpg'),
                        os.path.join(des_path, 'query', 'img', img_c + '.jpg'))
            shutil.copy(os.path.join(source_dir, 'mask', img_c + '.png'),
                        os.path.join(des_path, 'query', 'mask', img_c + '.png'))
        chunk_list = [num]
        middle_slice = None
