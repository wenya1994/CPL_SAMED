import os
import sys

import torch
import scipy.io
import scipy.io as sio

class Logger(object):
  def __init__(self, fpath=None):
    self.console = sys.stdout
    self.file = None
    if fpath is not None:
      self.file = open(fpath, 'w')

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
        self.file.write(msg)

  def flush(self):
    self.console.flush()
    if self.file is not None:
        self.file.flush()
        os.fsync(self.file.fileno())

  def close(self):
    self.console.close()
    if self.file is not None:
        self.file.close()

class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


import sys

from PIL import Image,ImageOps
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2
from torchvision.utils import save_image
from torchvision.transforms import ToTensor


def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    mx = max(img.size)
    mask = Image.new('RGB', (mx, mx), (0, 0, 0))
    mask.paste(img, (0, 0))  # 将 img 黏贴到 mask 黑色背景上
    mask = mask.resize(size)
    return mask


def rotate_images_in_directory(input_dir, output_dir):
    # 确保输出目录存在

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(input_dir, filename)

            # 打开图片
            with Image.open(filepath) as img:
                width, height = img.size
                # 根据图片尺寸决定是否旋转
                if width == 321 and height == 481:
                    img = img.rotate(90, expand=True)
                    print(f'Rotated {filename} to 481x321')
                elif width != 481 or height != 321:

                    print(f'Skipped {filename} because it has unexpected dimensions')
                # 保存图片到输出目录
                output_path = os.path.join(output_dir, filename)
                img.save(output_path)
                print(f'Saved {filename} to {output_dir}')

def caijian(folder_path):
    # 遍历文件夹中的所有文件

    for filename in os.listdir(folder_path):

        if filename.endswith('.jpg') or filename.endswith('.png'):  # 检查文件是否为图片

            img_path = os.path.join(folder_path, filename)

            # 打开图片

            with Image.open(img_path) as im:
                width, height = im.size

                # 裁剪图片，去除最后一行和最后一列

                cropped_im = im.crop((0, 0, width,height-1))
                #cropped_im = im.crop((0, 0, height - 1))

                # 保存裁剪后的图片

                cropped_im.save(os.path.join(folder_path, filename))


def keep_image_size_rgb(path, size=(256, 256)):
    img = Image.open(path)
    mx = max(img.size)
    mask = Image.new('RGB', (mx, mx), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask



def binarize_image(image, threshold=0.5):
    # 二值化矩阵,返回一个01矩阵
    return (image > threshold).astype(np.uint8)


# 读取图片所在文件夹，将图片名保存指lst文件中
def load_image_in_list(image_path):
    image_files = glob.glob(os.path.join(image_path, '*.jpg')) + \
                  glob.glob(os.path.join(image_path, '*.jpeg')) + \
                  glob.glob(os.path.join(image_path, '*.png'))
    with open('AS-OTC.lst', 'w') as lst_file:
        for image_file in image_files:
            # 写入图片文件的相对路径（相对于folder_path）
            lst_file.write(os.path.relpath(image_file, image_path) + '\n')


# 将图片按奇数行奇数列、奇数行偶数列、偶数行奇数列、偶数行偶数列进行分割
# input_folder为图片所在文件夹，lst_file_path为lst文件,output+folder为输出文件夹
def split_image_into_quadrants(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图片文件
    for image_name in os.listdir(input_folder):
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(input_folder, image_name)

            img = Image.open(image_path)
            # 将图像转化为数组
            img_array = np.array(img)
            # 彩色图和黑白图
            if len(img_array.shape) == 3:
                height, width, _ = img_array.shape
            else:
                height, width = img_array.shape
            # 分割图像
            odd_rows_odd_cols = img_array[::2, ::2]
            odd_rows_even_cols = img_array[::2, 1::2]
            even_rows_odd_cols = img_array[1::2, ::2]
            even_rows_even_cols = img_array[1::2, 1::2]

            # 将numpy数组转化为图像
            img_odd_odd = Image.fromarray(odd_rows_odd_cols)
            img_odd_even = Image.fromarray(odd_rows_even_cols)
            img_even_odd = Image.fromarray(even_rows_odd_cols)
            img_even_even = Image.fromarray(even_rows_even_cols)

            # 保存分割后的图像
            base_name, ext = os.path.splitext(image_name)
            img_odd_odd.save(os.path.join(output_folder, f"{base_name}_a{ext}"))
            img_odd_even.save(os.path.join(output_folder, f"{base_name}_b{ext}"))
            img_even_odd.save(os.path.join(output_folder, f"{base_name}_c{ext}"))
            img_even_even.save(os.path.join(output_folder, f"{base_name}_d{ext}"))

def is_edge(img_array):  # 判断该像素是否是边缘
    height, width = img_array.shape
    mask = np.zeros((height, width))
    mask[:height, :width] = img_array
    edges = np.zeros((img_array.shape[0]-1, img_array.shape[1]-1), dtype=int)
    # print(edges)
    for i in range(0, img_array.shape[0]-1):

        for j in range(0, img_array.shape[1]-1):
            window = mask[i:i + 2, j:j + 2]

            # 计算2x2区域内非零元素的数量
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 1 and non_zero_count != 4 and non_zero_count != 0:
                edges[i, j] = 1
            else:
                edges[i, j] = 0
    edges_uint8 = edges.astype(np.uint8)
    # edges_image = Image.fromarray(edges_uint8)
    return edges  # 返回的是图片的numpy数组



def iss_edge(img_array):  # 判断该像素是否是边缘
    height, width = img_array.shape
    mask = np.zeros((height, width))
    mask[:height, :width] = img_array
    edges = np.zeros((img_array.shape[0]-1, img_array.shape[1]-1), dtype=int)
    # print(edges)
    for i in range(0, img_array.shape[0]-1):

        for j in range(0, img_array.shape[1]-1):
            window = mask[i:i + 2, j:j + 2]

            # 计算2x2区域内非零元素的数量
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 1 and non_zero_count != 4 and non_zero_count != 0:
                edges[i, j] = 1
            else:
                edges[i, j] = 0
    edges_uint8 = edges.astype(np.uint8)
    # edges_image = Image.fromarray(edges_uint8)
    return edges  # 返回的是图片的numpy数组

def is_zituedge(img_array):
    height, width = img_array.shape
    edges1 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges2 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges3 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges4 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i-1, j-1]
            top_right = img_array[i-1, j+1]
            bottom_left = img_array[i+1, j-1]
            bottom_right = img_array[i+1, j+1]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 1 and non_zero_count != 4 and non_zero_count != 0:
                edges1[i-1, j-1] = 1
            else:
                edges1[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i-1, j]
            top_right = img_array[i+1, j]
            bottom_left = img_array[i-1, j+2]
            bottom_right = img_array[i+1, j+2]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 255 and non_zero_count != 4 and non_zero_count != 0:
                edges2[i-1, j-1] = 1
            else:
                edges2[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i, j-1]
            top_right = img_array[i, j+1]
            bottom_left = img_array[i+2, j-1]
            bottom_right = img_array[i+2, j+1]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 255 and non_zero_count != 4 and non_zero_count != 0:
                edges3[i-1, j-1] = 1
            else:
                edges3[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i, j]
            top_right = img_array[i, j+2]
            bottom_left = img_array[i+2, j]
            bottom_right = img_array[i+2, j+2]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 255 and non_zero_count != 4 and non_zero_count != 0:
                edges4[i-1, j-1] = 1
            else:
                edges4[i-1, j-1] = 0
    return edges1,edges2,edges3,edges4

def is_include_edge(img_array):  # 判断该像素围成的2*2区域是否包含边缘
    height, width = img_array.shape
    mask = np.zeros((height + 1, width + 1))
    mask[:height, :width] = img_array
    edges = np.zeros_like(img_array, dtype=int)
    # print(edges)
    for i in range(0, img_array.shape[0]):

        for j in range(0, img_array.shape[1]):
            window = mask[i:i + 2, j:j + 2]

            # 计算2x2区域内非零元素的数量
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if non_zero_count != 4 and non_zero_count != 0:
                edges[i, j] = 255
    return edges  # 返回的是以该像素为左上角的2*2区域是否包含边缘


# 一个转4个
# 传入png


#图片平均分四份  输入为图片文件夹和输出文件夹
def split_and_save_images(input_folder, output_folder):
    # 创建输出文件夹，如果它还不存在的话

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件

    for filename in os.listdir(input_folder):

        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # 构建完整的文件路径
            file_path = os.path.join(input_folder, filename)

            # 打开图像
            with Image.open(file_path) as img:
                # 获取图像尺寸
                width, height = img.size
                # 计算分割点
                w_split = width // 2
                h_split = height // 2
                # 分割并保存图像
                # 左上角
                box = (0, 0, w_split, h_split)
                cropped_img = img.crop(box)
                cropped_img.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_1.png"))

                # 左下角
                box = (0, h_split, w_split, height)
                cropped_img = img.crop(box)
                cropped_img.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_2.png"))
                # 右上角
                box = (w_split, 0, width, h_split)
                cropped_img = img.crop(box)
                cropped_img.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_3.png"))
                # 右下角
                box = (w_split, h_split, width, height)
                cropped_img = img.crop(box)
                cropped_img.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_4.png"))


def image_to_txt(image_path, txt_path):    #输入为图片 和txt文件位置
    # 打开图像并转换为灰度图像
    image = Image.open(image_path).convert('L')

    # 获取图像尺寸
    width, height = image.size

    # 将图像转换为numpy数组
    image_array = np.array(image)

    # 打开一个文件以写入
    with open(txt_path, 'w') as f:
        # 写入图像尺寸
        f.write(f"{width} {height}\n")

        # 写入像素值
        for row in image_array:
            row_str = ' '.join(map(str, row))
            f.write(row_str + '\n')

#将象限图拼回原图  输入：图片所在文件夹和输出文件夹
def combine_fourimages(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    grouped_files = {}
    # 筛选出符合命名规范的图像文件
    files = [f for f in os.listdir(input_folder) if f.endswith(('_1.png', '_2.png', '_3.png', '_4.png'))]

    # 根据基本名称组织文件
    for file in files:
        base_name = file.rsplit('_', 1)[0]
        suffix = file[-5]  # 获取后缀数字 '1', '2', '3', 或 '4'a
        if base_name not in grouped_files:
            grouped_files[base_name] = {}
        grouped_files[base_name][suffix] = file

    # 检查并组合图像
    for base_name, parts in grouped_files.items():
        required_parts = {'1', '2', '3', '4'}
        missing_parts = required_parts - set(parts.keys())
        if not missing_parts:
            # 按顺序加载图像：左上角（1），左下角（2），右上角（3），右下角（4）
            ordered_suffixes = ['1', '2', '3', '4']
            imgs = [np.array(Image.open(os.path.join(input_folder, parts[suffix]))) for suffix in ordered_suffixes]

            # 检测图像通道数
            channels = imgs[0].shape[2] if len(imgs[0].shape) == 3 else 1
            full_height, full_width = imgs[0].shape[0] * 2, imgs[0].shape[1] * 2
            if channels > 1:
                combined_image = np.zeros((full_height, full_width, channels), dtype=imgs[0].dtype)
            else:
                combined_image = np.zeros((full_height, full_width), dtype=imgs[0].dtype)

            # 组合图像
            combined_image[:full_height // 2, :full_width // 2] = imgs[0]  # 1 - 左上角
            combined_image[full_height // 2:, :full_width // 2] = imgs[1]  # 2 - 左下角
            combined_image[:full_height // 2, full_width // 2:] = imgs[2]  # 3 - 右上角
            combined_image[full_height // 2:, full_width // 2:] = imgs[3]  # 4 - 右下角

            # 保存组合后的图像
            output_image_path = os.path.join(output_folder, base_name + '.png')
            Image.fromarray(combined_image).save(output_image_path)
            #print(f"Combined image saved to {output_image_path}")
        else:
            print(f"Not all parts are available for {base_name}, missing: {', '.join(missing_parts)}")

#将奇偶图拼回原图并加一圈像素  输入：图片所在文件夹和输出文件夹
def combine_images_with_border(input_folder, output_folder, border_size=1, border_color='black'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    grouped_files = {}
    # 只读取符合指定后缀的文件
    files = [f for f in os.listdir(input_folder) if f.endswith(('_a.png', '_b.png', '_c.png', '_d.png'))]

    # 根据基本名称组织文件
    for file in files:
        base_name = file.rsplit('_', 1)[0]
        suffix = file[-5]  # 获取后缀字符 'a', 'b', 'c', 或 'd'
        if base_name not in grouped_files:
            grouped_files[base_name] = {}
        grouped_files[base_name][suffix] = file

    # 检查并组合图像
    for base_name, parts in grouped_files.items():
        required_parts = {'a', 'b', 'c', 'd'}
        missing_parts = required_parts - set(parts.keys())
        if not missing_parts:
            imgs = [np.array(Image.open(os.path.join(input_folder, parts[suffix]))) for suffix in sorted(parts)]

            # 检测图像通道数
            channels = imgs[0].shape[2] if len(imgs[0].shape) == 3 else 1
            full_height, full_width = imgs[0].shape[0] * 2, imgs[0].shape[1] * 2
            if channels > 1:
                combined_image = np.zeros((full_height, full_width, channels), dtype=imgs[0].dtype)
            else:
                combined_image = np.zeros((full_height, full_width), dtype=imgs[0].dtype)

            # 组合图像
            combined_image[::2, ::2] = imgs[0]  # a
            combined_image[::2, 1::2] = imgs[1]  # b
            combined_image[1::2, ::2] = imgs[2]  # c
            combined_image[1::2, 1::2] = imgs[3]  # d

            # 转换为Image对象并添加边框
            combined_image_pil = Image.fromarray(combined_image)


            combined_image_with_border = ImageOps.expand(combined_image_pil, border=border_size, fill=border_color)



            # 保存组合后的图像
            output_image_path = os.path.join(output_folder, base_name + '.png')
            combined_image_with_border.save(output_image_path)
            #print(f"Combined image with border saved to {output_image_path}")
        else:
            print(f"Not all parts are available for {base_name}, missing: {', '.join(missing_parts)}")


#def read_bsds_mat2png():



def create_lst_file(folder1, folder2, output_lst):
    # 读取第一个文件夹中的图片名
    images_train = [f"NYUD/train/image/{filename}" for filename in os.listdir(folder1) if filename.endswith(('.jpg', '.png'))]

    # 读取第二个文件夹中的图片名
    ground_truth_1th = [f"NYUD/train/GT/{filename}" for filename in os.listdir(folder2) if filename.startswith('1th') and filename.endswith(('.jpg', '.png'))]
    ground_truth_2th = [f"NYUD/train/GT/{filename}" for filename in os.listdir(folder2) if filename.startswith('2th') and filename.endswith(('.jpg', '.png'))]

    # 确保两列图片数目一致
    max_len = max(len(images_train), len(ground_truth_1th), len(ground_truth_2th))
    images_train += [""] * (max_len - len(images_train))
    ground_truth_1th += [""] * (max_len - len(ground_truth_1th))
    ground_truth_2th += [""] * (max_len - len(ground_truth_2th))

    # 将数据写入.lst文件
    with open(output_lst, 'w') as f:
        for img, gt1, gt2 in zip(images_train, ground_truth_1th, ground_truth_2th):
            f.write(f"{img} {gt1} {gt2}\n")



def is_zituedge(img_array):
    height, width = img_array.shape
    edges1 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges2 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges3 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges4 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i-1, j-1]
            top_right = img_array[i-1, j+1]
            bottom_left = img_array[i+1, j-1]
            bottom_right = img_array[i+1, j+1]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 1 and non_zero_count != 4 and non_zero_count != 0:
                edges1[i-1, j-1] = 1
            else:
                edges1[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i-1, j]
            top_right = img_array[i+1, j]
            bottom_left = img_array[i-1, j+2]
            bottom_right = img_array[i+1, j+2]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 1 and non_zero_count != 4 and non_zero_count != 0:
                edges2[i-1, j-1] = 1
            else:
                edges2[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i, j-1]
            top_right = img_array[i, j+1]
            bottom_left = img_array[i+2, j-1]
            bottom_right = img_array[i+2, j+1]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 1 and non_zero_count != 4 and non_zero_count != 0:
                edges3[i-1, j-1] = 1
            else:
                edges3[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i, j]
            top_right = img_array[i, j+2]
            bottom_left = img_array[i+2, j]
            bottom_right = img_array[i+2, j+2]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 1 and non_zero_count != 4 and non_zero_count != 0:
                edges4[i-1, j-1] = 1
            else:
                edges4[i-1, j-1] = 0
    return edges1,edges2,edges3,edges4



def iss_zituedge(img_array):
    height, width = img_array.shape
    edges1 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges2 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges3 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges4 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i-1, j-1]
            top_right = img_array[i-1, j+1]
            bottom_left = img_array[i+1, j-1]
            bottom_right = img_array[i+1, j+1]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 255 and non_zero_count != 4 and non_zero_count != 0:
                edges1[i-1, j-1] = 1
            else:
                edges1[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i-1, j]
            top_right = img_array[i+1, j]
            bottom_left = img_array[i-1, j+2]
            bottom_right = img_array[i+1, j+2]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 255 and non_zero_count != 4 and non_zero_count != 0:
                edges2[i-1, j-1] = 1
            else:
                edges2[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i, j-1]
            top_right = img_array[i, j+1]
            bottom_left = img_array[i+2, j-1]
            bottom_right = img_array[i+2, j+1]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 255 and non_zero_count != 4 and non_zero_count != 0:
                edges3[i-1, j-1] = 1
            else:
                edges3[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i, j]
            top_right = img_array[i, j+2]
            bottom_left = img_array[i+2, j]
            bottom_right = img_array[i+2, j+2]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 255 and non_zero_count != 4 and non_zero_count != 0:
                edges4[i-1, j-1] = 1
            else:
                edges4[i-1, j-1] = 0
    return edges1,edges2,edges3,edges4

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

#奇偶分四个数组
def split_array_into_oddeven(img_array):

    odd_rows_odd_cols = img_array[::2, ::2]
    odd_rows_even_cols = img_array[::2, 1::2]
    even_rows_odd_cols = img_array[1::2, ::2]
    even_rows_even_cols = img_array[1::2, 1::2]

    return odd_rows_odd_cols,odd_rows_even_cols,even_rows_odd_cols,even_rows_even_cols

def split_array_into_quarters(array):
    # 获取数组的行数和列数
    rows, cols = array.shape

    # 计算行和列的中点
    mid_row = rows // 2
    mid_col = cols // 2

    # 分割数组
    top_left = array[:mid_row, :mid_col]
    bottom_left = array[mid_row:, :mid_col]
    top_right = array[:mid_row, mid_col:]
    bottom_right = array[mid_row:, mid_col:]

    return top_left, bottom_left, top_right, bottom_right


def combine_arrays_into_tensor(*arrays):
    # 检查输入数组的数量
    if len(arrays) != 8:
        raise ValueError("必须提供8个数组")

    # 检查每个数组的形状
    # for array in arrays:
    #     if array.shape != (160, 256):
    #         raise ValueError("每个数组的形状必须为(160, 256)")

    # 将数组添加新的轴，变成(1, 160, 256)的形状
    reshaped_arrays = [array[np.newaxis, :, :] for array in arrays]

    # 将这些数组组合成一个张量，形状为(8, 1, 160, 256)
    tensor = np.stack(reshaped_arrays, axis=0)

    return tensor


def add_black_border(source_folder, target_folder, border_width=1, border_height=1):

    """

    Add a black border to the right and bottom of images in the source folder and save the result in the target folder.


    :param source_folder: Path to the folder containing the source images.

    :param target_folder: Path to save the modified images.

    :param border_width: Width of the border to be added on the right side.

    :param border_height: Height of the border to be added at the bottom.

    """

    # Ensure the target folder exists

    if not os.path.exists(target_folder):

        os.makedirs(target_folder)


    # Loop through all files in the source folder

    for filename in os.listdir(source_folder):

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):

            # Open the image

            img_path = os.path.join(source_folder, filename)

            img = Image.open(img_path).convert('L')  # Convert to grayscale if necessary


            # Get the original size of the image

            width, height = img.size


            # Create a new image with the increased size

            new_img = Image.new('L', (width + border_width, height + border_height), color=0)  # Black color


            # Paste the original image into the new image, leaving the border

            new_img.paste(img, (0, 0))


            # Save the new image

            output_path = os.path.join(target_folder, filename)

            new_img.save(output_path)

            print(f"Processed and saved: {output_path}")


def rename_images_in_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"文件夹 {folder_path} 不存在")

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 只处理图片文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            old_file_path = os.path.join(folder_path, filename)
            new_filename = '2th' + filename
            new_file_path = os.path.join(folder_path, new_filename)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"{filename} 已重命名为 {new_filename}")
# Usage example


def mat_to_png(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for mat_file in os.listdir(input_folder):

        if mat_file.endswith('.mat'):

            mat_path = os.path.join(input_folder, mat_file)

            mat_data = scipy.io.loadmat(mat_path)

            # 提取感兴趣的数据（例如 'groundTruth'）

            if 'groundTruth' in mat_data:

                mat_array = mat_data['groundTruth']

                # 如果数据结构复杂，你可能需要进一步提取

                if isinstance(mat_array, np.ndarray) and mat_array.size > 0:

                    # 假设 mat_array 是一个包含多个元素的数组

                    # 每个元素可能是一个包含多个字段的结构化数组

                    # 我们需要提取 'Segmentation' 或 'Boundaries' 字段

                    segmentations = []

                    for item in mat_array:

                        # 尝试访问 'Segmentation' 字段

                        try:

                            segmentation = item[0]['Segmentation'].squeeze()

                            if isinstance(segmentation, np.ndarray):
                                segmentations.append(segmentation)

                        except KeyError:

                            # 如果没有 'Segmentation' 字段，尝试访问 'Boundaries' 字段

                            try:

                                boundaries = item[0]['Boundaries'].squeeze()

                                if isinstance(boundaries, np.ndarray):
                                    segmentations.append(boundaries)

                            except KeyError:

                                print(f"No valid field found in item: {item}")

                                continue

                    # 假设 segmentations 中只有一个元素，取第一个元素

                    if segmentations:

                        segmentation = segmentations[0]

                        # 归一化处理

                        # 确保数组中没有 NaN 或 Inf 值

                        if np.isnan(segmentation).any() or np.isinf(segmentation).any():
                            print("Array contains NaN or Inf values.")

                            continue

                        # 归一化

                        if segmentation.dtype == np.uint16:

                            segmentation = segmentation.astype(np.float32)  # 避免溢出

                            normalized_segmentation = (segmentation - np.min(segmentation)) / (
                                        np.max(segmentation) - np.min(segmentation)) * 255

                            normalized_segmentation = normalized_segmentation.astype(np.uint8)

                        else:

                            # 直接归一化

                            normalized_segmentation = (segmentation - np.min(segmentation)) / (
                                        np.max(segmentation) - np.min(segmentation)) * 255

                            normalized_segmentation = normalized_segmentation.astype(np.uint8)

                        # 保存为 PNG

                        img = Image.fromarray(normalized_segmentation)

                        img_name = os.path.splitext(mat_file)[0] + '.png'

                        img.save(os.path.join(output_folder, img_name))


def process_mat_files(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有 .mat 文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mat"):
            file_path = os.path.join(input_folder, file_name)

            # 读取 .mat 文件
            mat_file = scipy.io.loadmat(file_path)

            # 提取 'Boundaries' 数据
            edge_image = mat_file['edge_gt']

            # 归一化图像到 0-255
            edge_image = (edge_image - np.min(edge_image)) / (np.max(edge_image) - np.min(edge_image)) * 255
            edge_image = edge_image.astype(np.uint8)

            # 将 numpy 数组转换为 PIL 图像对象
            img = Image.fromarray(edge_image)

            # 保存为 PNG 文件
            output_file_name = file_name.replace('.mat', '.png')
            output_path = os.path.join(output_folder, output_file_name)
            img.save(output_path)

            print(f"转换完成: {output_file_name}")


def load_and_print_mat_keys(file_path):
    # 加载MAT文件

    mat_data = scipy.io.loadmat(file_path)

    # 打印所有键值对

    for key in mat_data.keys():

        # 忽略系统的两个默认键 '__header__' 和 '__version__'

        if key not in ['__header__', '__version__', '__globals__']:
            print(f"键名: {key}, 值: {mat_data[key]}")


def rename_files_in_directory(directory_path):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        # 拼接文件的完整路径
        old_file_path = os.path.join(directory_path, filename)

        # 确保是文件而不是目录
        if os.path.isfile(old_file_path):
            # 获取文件名和扩展名
            name, ext = os.path.splitext(filename)

            # 确保文件名长度足够
            if len(name) >= 17:
                # 生成新的文件名
                new_name = name[:3] + name[-14:] + ext

                # 拼接新文件的完整路径
                new_file_path = os.path.join(directory_path, new_name)

                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} -> {new_file_path}")
            else:
                print(f"Filename '{filename}' is too short to rename.")


def load_and_stack_images(folder_path, target_shape=(160, 256)):
    """
    读取文件夹中的二值黑白图片并将其堆叠成一个 PyTorch 张量。

    参数:
        folder_path (str): 图片文件夹路径
        target_shape (tuple): 目标图片的形状 (height, width)

    返回:
        torch.Tensor: 形状为 [num_images, height, width] 的张量
    """
    # 获取文件夹中的所有文件
    image_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.bmp'))])

    if len(image_files) != 1600:
        raise ValueError(f"文件夹中图片数量不是1600张，当前数量为: {len(image_files)}")

    # 初始化一个空的列表来存储图片数据
    image_list = []

    for image_file in image_files:
        # 读取图片并转换为灰度图
        img = Image.open(image_file).convert('L')  # 'L' 表示灰度图
        # 调整图片大小为目标形状
        img = img.resize((target_shape[1], target_shape[0]), Image.NEAREST)  # NEAREST 保持二值特性
        # 将图片转换为 NumPy 数组
        img_array = np.array(img)
        # 归一化到 [0, 1] 范围（如果图片值是0和255）
        img_array = img_array / 255.0
        # 添加到列表中
        image_list.append(img_array)

    # 将列表转换为 NumPy 数组
    image_stack = np.stack(image_list, axis=0)
    # 转换为 PyTorch 张量
    image_tensor = torch.tensor(image_stack, dtype=torch.float32)

    return image_tensor


if __name__ == "__main__":
    folder_path = r"E:\UAED\6.UAED\6.UAED\result\bsds\round2\a1+a2\mask_a11"  # 替换为你的文件夹路径
    image_tensor = load_and_stack_images(folder_path)
    print(image_tensor.shape)  # 输出: torch.Size([1600, 160, 256]






