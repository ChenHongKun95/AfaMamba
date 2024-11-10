# AfaMamba
Official implementation of the paper  AfaMamba: Adaptive Feature Aggregation with Visual State Space Model for Remote Sensing Images Semantic Segmentation
# Introduction
AfaMamba based on GeoSeg, which is an open-source semantic segmentation toolbox based on PyTorch, pytorch lightning and timm, which mainly focuses on developing advanced for remote sensing image segmentation.
# Datasets
 * ISPRS Potsdam
 * LoveDA
 - Supported Remote Sensing Datasets
  - [ISPRS Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) 
  - [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)
    
# Folder Structure
Prepare the following folders to organize the data:
```none
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (Merge Train and Val)
│   ├── potsdam
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)

```
 
## Acknowledgement

Many thanks the following projects's contributions to **AfaMamba**.
- [GeoSeg](https://github.com/WangLibo1995/GeoSeg)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery](https://authors.elsevier.com/a/1fIji3I9x1j9Fs)
- [MambaIR: A Simple Baseline for Image Restoration with State-Space Model](https://github.com/csguoh/MambaIR)
- [Learning Spatial Fusion for Single-Shot Object Detection].(https://github.com/GOATmessi8/ASFF)
