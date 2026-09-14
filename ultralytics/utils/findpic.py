import os
import shutil

txt_path = 'D:\\underwater_datasets\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\ImageSets\\Main\\val.txt'
# 读取包含图片编号的txt文件
with open(txt_path, 'r') as f:  # 替换为你的txt文件路径
    target_numbers = {line.strip() for line in f}

# 设置路径
source_folder = 'D:\\underwater_datasets\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\Annotations'  # 原始图片存放目录
destination_folder = 'D:\\underwater_datasets\\VOCtrainval_06-Nov-2007\\VOCdevkit\\labels\\test'  # 目标存放目录

# 创建目标文件夹（如果不存在）
os.makedirs(destination_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 分割文件名和扩展名
    file_base, file_ext = os.path.splitext(filename)

    # 检查是否符合条件：扩展名为.jpg且文件名在目标编号集合中
    if file_ext.lower() == '.xml' and file_base in target_numbers:
        # 构建完整文件路径
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, filename)

        # 移动文件
        shutil.copy(src_path, dst_path)
        print(f'已复制: {filename}')

print("文件移动完成！")