import os
import shutil
import argparse
import yaml

config_file = '/root/SAM_Adapter_MAML/configs/cod-sam-vit-b.yaml'
with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# config = {'data_path': r'F:\AI_projects\Data'}
test_percent = 0.2
for dirpath, dirnames, filenames in os.walk(config['data_path']):
    print(dirpath.split('/')[-1])
    if 'train' in dirnames and 'test' not in dirnames:
        os.makedirs(os.path.join(dirpath, 'test', 'img'))
        os.makedirs(os.path.join(dirpath, 'test', 'mask'))
        for i, (img, mask) in enumerate(zip(os.listdir(os.path.join(dirpath, 'eval', 'img')),
                                            os.listdir(os.path.join(dirpath, 'eval', 'mask')))):
            shutil.move(os.path.join(dirpath, 'eval', 'img', img), os.path.join(dirpath, 'test', 'img', img))
            shutil.move(os.path.join(dirpath, 'eval', 'mask', mask), os.path.join(dirpath, 'test', 'mask', mask))
            if i > test_percent * len(os.listdir(os.path.join(dirpath, 'eval', 'img'))):
                break

print('done')
