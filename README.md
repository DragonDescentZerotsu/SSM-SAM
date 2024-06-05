# SSM-SAM
Code for WACV 2024, Self-Sampling Meta SAM: Enhancing Few-shot Medical Image Segmentation with Meta-Learning  

Due to the limited time in my summer research and very close deadline of WACV 2024, this project is in a hurry and the code is in chaos(sorry for that)  

If you try to run this project, refer to the file **train_MAML_new_bs4.py**, whose name means "the latest version of trainning code, and the batch size(bs) is 4". **Here, a batch contains all the slices in a chunk**. The one named **train_MAML_new_bs2.py** is for MRI modality as the slices are further away from each other, so one chunk of MRI images includes less slices than CT images do.  

Chunking CT / MRI scans is implemented in a very silly way as I don't want to change the code of previous dataloader. You have to shape your dataset like this (batch-size is 4, number of images in a chunk is 5):  

e.g. ------
<pre>
Dataset

--liver  
--spleen  
  --support_<strong>eval</strong>  
    --img  
      --3.jpg  
      --3.jpg   
      --3.jpg  
      --3.jpg  
      ------------  (virtual divider, do not really exist in the folder)
      --8.jpg
      ...
    --mask  
      --3.png  
      --3.png  
      --3.png  
      --3.png  
      ------------
      --8.png
      ...
  --query_<strong>eval</strong>  
    --img  
      --1.jpg  
      --2.jpg (<strong>notice</strong> : slice no.3 and no.8 are used as the support image in the chunk)  
      --4.jpg  
      --5.jpg  
      ------------
      --6.jpg
      --7.jpg
      --9.jpg
      --10.jpg
      ------------
      ...
    --mask  
      --1.png  
      --2.png (<strong>notice</strong> : slice no.3 and no.8 are used as the support image in the chunk)  
      --4.png  
      --5.png 
      ------------
      --6.jpg
      --7.jpg
      --9.jpg
      --10.jpg
      ------------
      ...
  --support_train  
    --(the same as evaluation(eval) data organization)
  --query_train  
    --(the same as evaluation(eval) data organization)
--IVC  
--stomach
</pre>  

During training and inference, the four support masks in one batch (actually the same mask) will be loaded together with the four query images in the same chunk to train the model or do inference.  

