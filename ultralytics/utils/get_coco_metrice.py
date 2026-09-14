import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os

# def increment_image_id(input_file, output_file):
#     try:
#         with open(input_file, 'r') as f:
#             data = json.load(f)
#
#         for item in data:
#             item['category_id'] += 1
#
#         with open(output_file, 'w') as f:
#             json.dump(data, f, indent=4)
#
#         print(f"已将 {input_file} 中的 category_id 加 1，并保存到 {output_file}。")
#     except FileNotFoundError:
#         print(f"错误：未找到 {input_file} 文件。")
#     except json.JSONDecodeError:
#         print(f"错误：{input_file} 不是有效的 JSON 文件。")
#     except Exception as e:
#         print(f"发生未知错误：{e}")

#
# if __name__ == "__main__":
#     input_file = '/home0/students/master/2024/liuyj/learn/yan_mmcv/ultralytics-yolov11_new/runs/val/RUOD+LittleAttn_coco2/predictions.json'
#     output_file = '/home0/students/master/2024/liuyj/learn/yan_mmcv/ultralytics-yolov11_new/runs/val/RUOD+LittleAttn_coco2/newPre.json'
#     increment_image_id(input_file, output_file)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str,
                        default='/home/students/master/2024/liuyj/learn/Dataset/DUO/annotations/instances_val2017.json',
                        help='training model path')
    parser.add_argument('--pred_json', type=str,
                        default='/home/students/master/2024/liuyj/learn/paper2/ultralytics-yolov11/runs/val/2025-10-23/newSpatialAttn+DUO/predictions.json',
                        help='data yaml path')

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

