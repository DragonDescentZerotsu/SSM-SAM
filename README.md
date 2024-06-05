# SSM-SAM
Code for WACV 2024, Self-Sampling Meta SAM: Enhancing Few-shot Medical Image Segmentation with Meta-Learning  

Due to the limited time in my summer research and very close deadline of WACV 2024, this project is in a hurry and the code is in chaos(sorry for that)  

If you try to run this project, refer to the file **train_MAML_new_bs4.py**, whose name means "the latest version of trainning code and the batch size(bs) is 4". The one named **train_MAML_new_bs2.py** is for MRI modality as the slices are further away from each other, so one chunk of MRI images includes less slices than CT images do.  

Chunking CT scans is implemented in a very silly way as I don't want to change the code of previous dataloader. You have to shape your dataset like this (batch-size is 4, number of images in a chunk is 5):  

e.g. ------

Dataset

--liver  
--spleen  
  --support_**eval**  
    --img  
      --3.jpg  
      --3.jpg   
      --3.jpg  
      --3.jpg  
    --mask  
      --3.png  
      --3.png  
      --3.png  
      --3.png  
  --query_**eval**  
    --img  
      --5.jpg  
      --4.jpg (**notice : slice no.3 is used as the support image in the chunk)  
      --2.jpg  
      --1.jpg  
    --mask  
      --5.png  
      --4.png (**notice : slice no.3 is used as the support image in the chunk)  
      --2.png  
      --1.png  
  --support_train  
  --query_train  
--IVC  
