# GCM-Net: Towards Effective Global Context Modeling for Image Inpainting (ACM MM2021)

# Pytorch 
Pytorch implementation of GCM-Net

## Environment
1. Python 3.6 
2. Pytorch 1.3.0
3. torchvision 0.4.1
4. cuda 11.4

### Train: 
`python main.py --bs batch_size --img_flist image_path --mask_flist mask_path --nEpochs 40 --lr 0.0001`

### Test
`python eval.py --bs batch_size --model checkpoint --img_flist image_path --mask_flist mask_path`

