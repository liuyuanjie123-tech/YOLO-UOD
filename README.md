# TGRS YOLO-UOD 2026
This is an official PyTorch implementation of paper [YOLO-UOD: Location-Edge Co-Perception Network With Multiscale Features Fusion for Underwater Object Detection](https://ieeexplore.ieee.org/abstract/document/11563894)

## Quick Start

This code repository includes the base run code for YOLO-UOD, see folder `./weights` for the weights file.

### 1. Deploy Conda environment
```Command Line
conda create -n YOLO-UOD python==3.8
```

### 2. Install package dependencies
```Command Line
pip install -r requirements.txt
```

### 3. Train Model (Optional, requires Datasets and Cuda)
The implementation of all our models is contained in the "ultralytics/nn/modules/base_layer.py" file. The dataset is configured in the "data" directory.
```Command Line
conda activate YOLO-UOD
python train.py
```

### 4. Test Model (Optional, requires Datasets and Cuda)
```Command Line
conda activate YOLO-UOD
python start_predict.py
```

### 5. Dectet
```Command Line
conda activate YOLO-UOD
python start_val.py
```

## Cite
You can cite our work in the following format:

```bibtex
@ARTICLE{11563894,
  author={Luo, Fulin and Liu, Yuanjie and Guo, Tan and Fu, Chuan and Lin, Yukun and Xiang, Tao and Du, Bo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={YOLO-UOD: Location-Edge Co-Perception Network With Multiscale Features Fusion for Underwater Object Detection}, 
  year={2026},
  volume={64},
  number={},
  pages={5627314-5627314},
  keywords={Modeling;YOLO;Object detection;Convolution;Modules (abstract algebra);Conferences;Computers;Frequency;Visualization;Computer vision;Attention mechanism;edge and location perception;multiscale feature;underwater object detection (UOD);wavelet convolution},
  doi={10.1109/TGRS.2026.3703991}}

```
