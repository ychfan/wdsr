# Wide Activation for Efficient Image and Video Super-Resolution
Reloaded PyTorch implementation of WDSR, *BMVC 2019* [[pdf]](https://bmvc2019.org/wp-content/uploads/papers/0288-paper.pdf).

[Previous Implementations](https://github.com/JiahuiYu/wdsr_ntire2018)

## Performance
Small models

| Networks | Parameters | DIV2K (val) | Set5 | B100 | Urban100 | Pre-trained | Eval cmd | Train cmd |
| - | - | - | - | - | - | - | - | - |
| WDSR x2 | 1,190,100 | 34.76 | 38.08 | 32.23 | 32.34 | [Download](https://github.com/ychfan/wdsr/files/4176974/wdsr_x2.zip) | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 2 --job_dir X --ckpt ./wdsr_x2/epoch_30.pth --eval_only```</details> | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 2 --job_dir ./wdsr_x2```</details> |
| WDSR x3 | 1,195,605 | 31.03 | 34.45 | 29.14 | 28.33 | [Download](https://github.com/ychfan/wdsr/files/4176981/wdsr_x3.zip) | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 3 --job_dir X --ckpt ./wdsr_x3/epoch_30.pth --eval_only```</details> | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 3 --job_dir ./wdsr_x3```</details> |
| WDSR x4 | 1,203,312 | 29.04 | 32.22 | 27.61 | 26.21 | [Download](https://github.com/ychfan/wdsr/files/4176985/wdsr_x4.zip) | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 4 --job_dir X --ckpt ./wdsr_x4/epoch_30.pth --eval_only```</details> | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 4 --job_dir ./wdsr_x4```</details> |

Large models

| Networks | Parameters | DIV2K (val) | Set5 | B100 | Urban100 | Pre-trained | Eval cmd | Train cmd |
| - | - | - | - | - | - | - | - | - |
| WDSR x2 | 37,808,180 | 35.06 | 38.28 | 32.38 | 33.07 | [Download](https://drive.google.com/file/d/10OsQD--qWZIBinFignAWwppHw5z9LPMI/view?usp=sharing) | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --num_blocks 32 --num_residual_units 128 --scale 2 --job_dir X --ckpt ./wdsr_x2/epoch_30.pth --eval_only```</details> | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --num_blocks 32 --num_residual_units 128 --scale 2 --job_dir ./wdsr_x2```</details> |
| WDSR x3 | 37,826,645 | 31.34 | 34.76 | 29.32 | 28.94 | [Download](https://drive.google.com/file/d/10Yh0mI2825k69vChRZRGMsAC7C-M5hbk/view?usp=sharing) | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --num_blocks 32 --num_residual_units 128 --scale 3 --job_dir X --ckpt ./wdsr_x3/epoch_30.pth --eval_only```</details> | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --num_blocks 32 --num_residual_units 128 --scale 3 --job_dir ./wdsr_x3```</details> |
| WDSR x4 | 37,852,496 | 29.33 | 32.58 | 27.78 | 26.79 | [Download](https://drive.google.com/file/d/10sYc5F63-o3eovtGCG5SSawk4otEHIxe/view?usp=sharing) | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --num_blocks 32 --num_residual_units 128 --scale 4 --job_dir X --ckpt ./wdsr_x4/epoch_30.pth --eval_only```</details> | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --num_blocks 32 --num_residual_units 128 --scale 4 --job_dir ./wdsr_x4```</details> |

## Usage

### Dependencies
```bash
conda install pytorch torchvision -c pytorch
conda install tensorboard h5py scikit-image
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git
```

### Evaluation

```bash
python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 2 --job_dir ./wdsr_x2 --eval_only
# or
python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 2 --job_dir ./wdsr_x2 --ckpt ./latest.pth --eval_only
```

## Datasets
[DIV2K dataset: DIVerse 2K resolution high quality images as used for the NTIRE challenge on super-resolution @ CVPR 2017](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

[Benchmarks (Set5, BSDS100, Urban100)](http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip)

Download and organize data like: 
```bash
wdsr/data/DIV2K/
├── DIV2K_train_HR
├── DIV2K_train_LR_bicubic
│   └── X2
│   └── X3
│   └── X4
├── DIV2K_valid_HR
└── DIV2K_valid_LR_bicubic
    └── X2
    └── X3
    └── X4
wdsr/data/Set5/*.png
wdsr/data/BSDS100/*.png
wdsr/data/Urban100/*.png
```
