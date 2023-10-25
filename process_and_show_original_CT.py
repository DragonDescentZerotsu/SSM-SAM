from PIL import Image, ImageEnhance
from skimage import transform
import numpy as np

#装载图像
im = Image.open('/Users/kirianozan/Documents/AI_project/Data/liver/train/img/11.jpg')

#裁剪图像
left = 0
top = 0
right = 480
bottom = 550
im_cropped = im.crop((left, top, right, bottom))

#进行错切变换
im_array = np.array(im_cropped)
tform = transform.AffineTransform(shear=-0.2) # 这里的0.2是错切强度，可以调整
im_shear = transform.warp(im_array, tform)
im_shear = Image.fromarray((im_shear * 255).astype(np.uint8))

#增加对比度
enhancer = ImageEnhance.Contrast(im_shear)
im_enhanced = enhancer.enhance(3) # 这里的1.5是对比度增加的倍数，可以调整

#保存修改后的图像
im_enhanced.save('/Users/kirianozan/Desktop/sample_transformed.jpg')