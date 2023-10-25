import os
import shutil

folder = '/root/SAM_Adapter_MAML/tensorboard'

# 删除目录及其所有内容
shutil.rmtree(folder)

# 重新创建一个同名的空目录
os.mkdir(folder)