import argparse
from ultralytics import YOLO
import os
from pathlib import Path
import sys

quiet = "quiet"
os.environ["GIT_PYTHON_REFRESH"] = quiet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_args():
    # 模型参数
    parser = argparse.ArgumentParser(description='Data Postprocess')
    parser.add_argument('--model', type=str, default=ROOT / 'runs/segment/train/2026-3-5/UIIS-PGA-MOE(w/o detail)(drop-cooldown-180)/weights/best.pt',
                        help='load the model')
    parser.add_argument('--data-dir', type=str, default=ROOT / 'data/DUO.yaml', help='the dir to data')
    parser.add_argument('--hyp', type=str, default=None, help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 多进程
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')

    # 训练中保存文件的地址
    parser.add_argument('--project', default=ROOT / 'runs/val/2025-11-24', help='save to project/name')
    parser.add_argument('--name', default='DUO+yolov10s', help='save to project/name')
    # 可以冻住前几层
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # comet_ml.init()
    model = YOLO(args.model)
    metrics = model.val(data=args.data_dir, batch=args.batch_size, cfg=args.hyp,
                        device=args.device,
                        project=args.project, name=args.name, seed=args.seed,
                        save_json=True,
                        plots=True,
                        workers=args.workers)


if __name__ == '__main__':
    main()
