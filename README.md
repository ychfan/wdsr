# Wide Activation for Efficient Image and Video Super-Resolution
Reloaded PyTorch implementation of WDSR, *BMVC 2019* [[pdf]](https://bmvc2019.org/wp-content/uploads/papers/0288-paper.pdf).

[Previous Implementations](https://github.com/JiahuiYu/wdsr_ntire2018)

## Performance
| Networks | Parameters | DIV2K (val) | Set5 | B100 | Urban100 | Pre-trained models | Training command |
| - | - | - | - | - | - | - | - |
| WDSR x2 | 1,190,100 | 34.76 | 38.08 | 32.23 | 32.34 | [Download](https://github.com/ychfan/wdsr/files/4176974/wdsr_x2.zip) |<details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 2 --job_dir ./wdsr_x2```</details> |
| WDSR x3 | 1,195,605 | 31.03 | 34.45 | 29.14 | 28.33 | [Download](https://github.com/ychfan/wdsr/files/4176981/wdsr_x3.zip) |<details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 3 --job_dir ./wdsr_x3```</details> |
| WDSR x4 | 1,203,312 | 29.04 | 32.22 | 27.61 | 26.21 | [Download](https://github.com/ychfan/wdsr/files/4176985/wdsr_x4.zip) |<details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 4 --job_dir ./wdsr_x4```</details> |

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
