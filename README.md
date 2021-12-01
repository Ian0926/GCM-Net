# GCM-Net: Towards Effective Global Context Modeling for Image Inpainting (ACM MM2021)

Official Pytorch implementation of [GCM-Net: Towards Effective Global Context Modeling for Image Inpainting](https://dl.acm.org/doi/pdf/10.1145/3474085.3475433).

## Environment:
1. Python 3.6 
2. Pytorch 1.3.0
3. torchvision 0.4.1
4. cuda 11.4

### Train: 
`python main.py --bs batch_size --img_flist image_path --mask_flist mask_path`

### Test:
`python eval.py --bs batch_size --model checkpoint --img_flist image_path --mask_flist mask_path --save`

### Bibtex:
```
@inproceedings{zheng2021gcm,
  title={GCM-Net: Towards Effective Global Context Modeling for Image Inpainting},
  author={Zheng, Huan and Zhang, Zhao and Wang, Yang and Zhang, Zheng and Xu, Mingliang and Yang, Yi and Wang, Meng},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={2586--2594},
  year={2021}
}
```
