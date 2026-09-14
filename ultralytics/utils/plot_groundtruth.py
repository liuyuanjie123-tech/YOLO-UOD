import numpy as np
import cv2
import os

# 定义类别名称（需与标签中的类别ID顺序一致）
CLASS_NAMES = ['holothurian', 'echinus', 'scallop', 'starfish']
# 使用BGR颜色（OpenCV标准），确保颜色定义清晰
COLORS = {
    0: (255, 40, 7),  # 蓝色
    1: (236, 219, 12),  # 浅灰色
    2: (235, 255, 238),  # 青色
    3: (185, 223, 3)  # 青绿色
}
TXT_COLORS = {
    0: (255, 213, 191),  # 蓝色
    1: (235, 255, 238),  # 浅灰色
    2: (235, 255, 238),  # 青色
    3: (235, 255, 238)  # 青绿色
}

# 定义深色和浅色集合（使用元组，因为集合元素必须是不可变的）
DARK_COLORS = {
    (235, 219, 11),  # 黄色
    (183, 223, 0),  # 绿色
    (221, 111, 255),  # 紫色
    (0, 237, 204),  # 青绿色
    (68, 243, 0),  # 亮绿色
    (255, 255, 0),  # 亮黄色
    (179, 255, 1),  # 浅绿色
    (11, 255, 162)  # 青绿色
}

LIGHT_COLORS = {
    (255, 42, 4),  # 红色
    (79, 68, 255),  # 蓝色
    (255, 0, 189),  # 粉色
    (255, 180, 0),  # 橙色
    (186, 0, 221),  # 紫色
    (0, 192, 38),  # 绿色
    (255, 36, 125),  # 粉红色
    (104, 0, 123),  # 深紫色
    (108, 27, 255),  # 紫色
    (47, 109, 252),  # 蓝色
    (104, 31, 17)  # 棕色
}


def hex2rgb(h):
    """将十六进制颜色转换为RGB值"""
    return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def get_txt_color(color, txt_color=(255, 255, 255)):
    """根据背景颜色确定文本颜色（BGR顺序）"""
    # 转换颜色为元组以便在集合中查找
    color_tuple = (int(color[0]), int(color[1]), int(color[2]))

    if color_tuple in DARK_COLORS:
        # 深色背景使用浅色文本（BGR顺序的白色）
        return (255, 255, 255)
    elif color_tuple in LIGHT_COLORS:
        # 浅色背景使用深色文本（BGR顺序的黑色）
        return (0, 0, 0)
    else:
        # 其他情况使用默认文本颜色
        return txt_color


def xywh2xyxy(x, img_width, img_height):
    """将YOLO格式的(x,y,w,h)转换为矩形框坐标(x1,y1,x2,y2)"""
    class_id = int(x[0])
    x_center, y_center, w, h = x[1:]
    x1 = max(0, (x_center - w / 2) * img_width)
    y1 = max(0, (y_center - h / 2) * img_height)
    x2 = min(img_width, (x_center + w / 2) * img_width)
    y2 = min(img_height, (y_center + h / 2) * img_height)
    return (int(x1), int(y1), int(x2), int(y2)), class_id


def plot_label_boxes(image_folder, label_folder, output_folder):
    """在图像上绘制标签边界框和类别名称"""
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图像 {img_path}")
            continue

        img_height, img_width = img.shape[:2]
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(label_folder, label_filename)

        if not os.path.exists(label_path):
            print(f"警告：标签文件 {label_path} 不存在，跳过")
            cv2.imwrite(os.path.join(output_folder, filename), img)
            continue

        with open(label_path, 'r') as f:
            labels = [line.strip().split() for line in f]
            if not labels:
                print(f"信息：标签文件 {label_path} 为空，保存原图")
                cv2.imwrite(os.path.join(output_folder, filename), img)
                continue

        for label in labels:
            try:
                label = np.array(label, dtype=np.float32)
                (x1, y1, x2, y2), class_id = xywh2xyxy(label, img_width, img_height)

                # 确保类别ID在COLORS字典范围内
                if class_id not in COLORS:
                    print(f"警告：类别ID {class_id} 没有对应的颜色定义，使用默认颜色")
                    color = (0, 255, 0)  # 默认绿色
                else:
                    color = COLORS[class_id]

                class_name = CLASS_NAMES[class_id]

                # 绘制矩形框
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 10)

                # 绘制类别名称（在框上方居中）
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                font_thickness = 4

                (text_width, text_height), _ = cv2.getTextSize(class_name, font, font_scale, font_thickness)
                text_x = x1 + (x2 - x1 - text_width) // 2

                # 确保文本不会超出图像顶部
                if y1 > text_height + 10:
                    text_y = y1 - 5
                else:
                    text_y = y1 + text_height + 10  # 放在框下方

                # 获取文本颜色
                txt_color = TXT_COLORS[class_id]

                # 绘制文本
                cv2.putText(img, class_name, (text_x, text_y),
                            font, font_scale, txt_color, font_thickness)
            except IndexError as e:
                print(f"错误：类别ID {class_id} 超出范围，检查标签类别定义 - {e}")
            except Exception as e:
                print(f"处理标签时发生错误: {e}")

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)
        print(f"已保存 {output_path}")


# 输入参数（用户提供）
image_folder = r'/home0/students/master/2024/liuyj/learn/Dataset/DUO/test_images/images/'
label_folder = r'/home0/students/master/2024/liuyj/learn/Dataset/DUO/test_images/labels/'
output_folder = r'/home0/students/master/2024/liuyj/learn/Dataset/DUO/test_images/plot_images'

# 执行绘制
plot_label_boxes(image_folder, label_folder, output_folder)