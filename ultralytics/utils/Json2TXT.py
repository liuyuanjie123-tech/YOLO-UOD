import json
import os


def json_to_yolo_txt(json_path, output_folder):
    # 读取 JSON 文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 处理 images 信息
    ids = []
    images = data.get('images', {})
    anns = data.get('annotations', {})
    for image_info in images:
        image_id = image_info['id']
        image_filename = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']

        # 处理 annotations
        annotations = [ann for ann in anns if ann['image_id'] == image_id]
        for annotation in annotations:
            bbox = annotation['bbox']
            category_id = int(annotation['category_id'])
            if category_id not in ids:
                ids.append(category_id)

            # 计算中心点坐标和宽高的相对值
            x_center = (bbox[0] + bbox[2] / 2) / image_width
            y_center = (bbox[1] + bbox[3] / 2) / image_height
            width = bbox[2] / image_width
            height = bbox[3] / image_height

            # 构造每个 annotation 对应的 TXT 文件内容
            txt_content = f"{category_id - 1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

            # 写入 TXT 文件
            txt_filename = f"{image_filename.replace('.jpg', '')}.txt"
            txt_path = os.path.join(output_folder, txt_filename)
            with open(txt_path, 'a') as txt_file:
                txt_file.write(txt_content)

    print("TXT 文件已生成！")


if __name__ == '__main__':
    # 指定输入的 JSON 文件和输出的文件夹路径
    json_file_path = "/home0/students/master/2024/liuyj/learn/Dataset/trashCan/annotations/instances_val_trashcan.json"
    output_folder_path = "/home0/students/master/2024/liuyj/learn/Dataset/trashCan/labels/test"

    # 调用函数进行转换
    json_to_yolo_txt(json_file_path, output_folder_path)
