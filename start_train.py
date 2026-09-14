import argparse
from ultralytics import YOLO
# import comet_ml
import os
from pathlib import Path
import sys
import warnings

quiet = "quiet"
os.environ["GIT_PYTHON_REFRESH"] = quiet
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

os.environ["WANDB_API_KEY"] = '9d31ed90714095b8dc8ebd418c0296091e288ec1'
os.environ["WANDB_MODE"] = "offline"
warnings.filterwarnings("ignore",
                        message="upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation")


def parse_args():
    # 模型参数
    parser = argparse.ArgumentParser(description='Data Postprocess')
    parser.add_argument('--model', type=str, default=ROOT / 'ultralytics/cfg/models/11/yolo11s-myModel.yaml',
                        help='load the model')
    parser.add_argument('--pretrained', type=str, default='',
                        help='(bool | str) whether to use a pretrained model (bool) or a model to load weights from (str)')
    parser.add_argument('--data-dir', type=str, default=ROOT / 'data/DUO.yaml', help='the dir to data')
    parser.add_argument('--hyp', type=str, default=None, help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--device', default='4', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # 多进程
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')

    # 训练中保存文件的地址
    parser.add_argument('--project', default=ROOT / 'runs/train/2026-9-14', help='save to project/name')
    parser.add_argument('--name', default='test', help='save to project/name')
    # 可以冻住前几层
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # comet_ml.init()
    model = YOLO(args.model)
    model.train(data=args.data_dir, epochs=args.epochs, batch=args.batch_size, cfg=args.hyp,
                device=args.device,
                imgsz=640,
                project=args.project, name=args.name, seed=args.seed,
                workers=args.workers,amp=False)


if __name__ == '__main__':
    main()
