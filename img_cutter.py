import os
import numpy as np
from PIL import Image
from pathlib import Path
import tifffile as tiff

# 定义切割尺寸
CROP_SIZE = 256  # 修改为128x128

def crop_images(input_dir, output_dir, data_types=["top", "topm", "labels"]):
    """
    将输入目录中的图像切割成指定大小的图像块
    
    参数:
    input_dir (str): 输入数据集根目录
    output_dir (str): 输出数据集根目录
    data_types (list): 需要处理的数据类型
    """
    # 遍历训练、验证和测试数据集
    for split in ["train_data", "valid_data", "test_data"]:
        split_input_dir = os.path.join(input_dir, split)
        split_output_dir = os.path.join(output_dir, split)
        
        # 创建输出目录
        for data_type in data_types:
            os.makedirs(os.path.join(split_output_dir, data_type), exist_ok=True)
        
        # 获取所有图像ID
        top_dir = os.path.join(split_input_dir, "top")
        image_ids = [f.split("_")[-1].replace(".tif", "") for f in os.listdir(top_dir) if f.endswith(".tif")]
        
        # 处理每个图像ID
        for image_id in image_ids:
            process_image(image_id, split_input_dir, split_output_dir, data_types)
    
    print("所有图像切割完成!")

def process_image(image_id, input_dir, output_dir, data_types):
    """
    处理单个图像及其对应的数据
    
    参数:
    image_id (str): 图像ID
    input_dir (str): 输入数据集目录
    output_dir (str): 输出数据集目录
    data_types (list): 需要处理的数据类型
    """
    # 读取TOP图像以获取尺寸信息
    top_path = os.path.join(input_dir, "top", f"top_mosaic_09cm_{image_id}.tif")
    top_img = tiff.imread(top_path)
    
    # 获取图像尺寸
    height, width = top_img.shape[:2]
    
    # 计算切割位置
    rows = height // CROP_SIZE
    cols = width // CROP_SIZE
    
    # 切割图像
    for row in range(rows):
        for col in range(cols):
            # 计算切割区域
            y1 = row * CROP_SIZE
            y2 = (row + 1) * CROP_SIZE
            x1 = col * CROP_SIZE
            x2 = (col + 1) * CROP_SIZE
            
            # 处理每种数据类型
            for data_type in data_types:
                # 构建输入文件路径
                if data_type == "top":
                    input_path = os.path.join(input_dir, data_type, f"top_mosaic_09cm_{image_id}.tif")
                elif data_type == "topm":
                    input_path = os.path.join(input_dir, data_type, f"dsm_09cm_matching_{image_id}.tif")
                else:  # gts_for_participants
                    input_path = os.path.join(input_dir, data_type, f"top_mosaic_09cm_{image_id}.tif")
                
                # 读取图像
                img = tiff.imread(input_path)
                
                # 切割图像
                if len(img.shape) == 3:  # 多通道图像
                    crop = img[y1:y2, x1:x2, :]
                else:  # 单通道图像
                    crop = img[y1:y2, x1:x2]
                
                # 构建输出文件路径
                output_filename = f"area{image_id}_row{row}_col{col}.tif"
                output_path = os.path.join(output_dir, data_type, output_filename)
                
                # 保存切割后的图像
                tiff.imwrite(output_path, crop)
    
    print(f"已处理图像 {image_id}，切割成 {rows*cols} 个图像块")

if __name__ == "__main__":
    # 设置输入和输出目录
    input_dir = "data_vaihin"  # 修改为上一步切割后的目录
    output_dir = "data_vaihin_cropped_256"  # 新的输出目录
    
    crop_images(input_dir, output_dir)    