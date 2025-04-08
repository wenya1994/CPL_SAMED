# import os, sys
#
# sys.path.append("/data/private/zhoucaixia/workspace/UD_Edge")
# import numpy as np
# from PIL import Image
# import cv2
# import argparse
# import time
# import torch
# import matplotlib
#
# matplotlib.use('Agg')
# from data.data_loader_one_random_uncert_yyz import BSDS_RCFLoader
# from models.sigma_logit_unetpp import Mymodel
# from torch.utils.data import DataLoader
#
# from os.path import join, split, isdir, splitext, split, abspath, dirname
# import scipy.io as io
# from torch.distributions import Normal, Independent
# from shutil import copyfile
# from PIL import Image
# from time import time  # Import time to measure execution time
#
# parser = argparse.ArgumentParser(description='PyTorch Training')
#
# parser.add_argument('--distribution', default="gs", type=str, help='the output distribution')
# args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# #test_dataset = BSDS_RCFLoader(root='./data_file/images/1to4test', split="test")
# test_dataset = BSDS_RCFLoader(root='./data_file', split="test")
# test_loader = DataLoader(
#     test_dataset, batch_size=1,
#     num_workers=0, drop_last=True, shuffle=False)
#
# with open('./data_file/test.lst', 'r') as f:
#     test_list = f.readlines()
# test_list = [split(i.rstrip())[1] for i in test_list]
# assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))
#
# model_path = "./Pth/epoch-16-checkpoint.pth"
# model = Mymodel(args)
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# #model.cuda()
# checkpoint = torch.load(model_path, map_location=device)
# model.load_state_dict(checkpoint['state_dict'])
# model.eval()
#
# save_dir = "./test/UAED"
#
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# first_ten_batches = list(test_loader)[:]
# start_time = time()  # Start timing
# for idx, image in enumerate(first_ten_batches):
#     filename = splitext(test_list[idx])[0]
#     image = image.cuda()
#     mean, std = model(image)
#     _, _, H, W = image.shape
#     outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
#     outputs = torch.sigmoid(outputs_dist.rsample())
#     png = torch.squeeze(1-outputs.detach()).cpu().numpy()
#     result = np.zeros((H , W ))
#     result[0:, 0:] = png
#     result_png = Image.fromarray((result * 255).astype(np.uint8))
#     png_save_dir = os.path.join(save_dir, "png")
#     mat_save_dir = os.path.join(save_dir, "mat")
#     os.makedirs(png_save_dir, exist_ok=True)
#     os.makedirs(mat_save_dir, exist_ok=True)
#     result_png.save(join(png_save_dir, "%s.png" % filename))
#     io.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)
# end_time = time()  # End timing
#
# elapsed_time = end_time - start_time
# print(len(test_loader))
# fps = len(test_loader) / elapsed_time
# print(f"Average FPS: {fps:.2f}")  # Print the average FPS
#
# import os
# from PIL import Image
#
#
# def crop_images_in_place(folder, crop_pixels=31):
#     """
#     Crop the right and bottom `crop_pixels` pixels of all images in the folder
#     and replace the original images with the cropped versions.
#
#     :param folder: Path to the folder containing images
#     :param crop_pixels: Number of pixels to crop from right and bottom
#     """
#     # Iterate over all files in the folder
#     for file_name in os.listdir(folder):
#         file_path = os.path.join(folder, file_name)
#
#         # Try opening the file as an image
#         try:
#             with Image.open(file_path) as img:
#                 # Calculate the new dimensions
#                 width, height = img.size
#                 cropped_width = width - crop_pixels
#                 cropped_height = height - crop_pixels
#
#                 # Perform the cropping
#                 cropped_img = img.crop((0, 0, cropped_width, cropped_height))
#
#                 # Save the cropped image, replacing the original
#                 cropped_img.save(file_path)
#
#                 print(f"Cropped and replaced: {file_name}")
#         except Exception as e:
#             print(f"Error processing {file_name}: {e}")
#
#
# # Specify your folder here
# folder = r"E:\6.UAED\test\UAED\png"
#
# # Call the function
# crop_images_in_place(folder)


'''
双卡
'''
import os, sys

sys.path.append("/data/private/zhoucaixia/workspace/UD_Edge")
import numpy as np
from PIL import Image
import cv2
import argparse
import time
import torch
import matplotlib

matplotlib.use('Agg')
from data.data_loader_one_random_uncert_yyz import BSDS_RCFLoader
from models.sigma_logit_unetpp import Mymodel
from torch.utils.data import DataLoader

from os.path import join, split, isdir, splitext, split, abspath, dirname
import scipy.io as io
from torch.distributions import Normal, Independent
from shutil import copyfile
from PIL import Image
from time import time  # Import time to measure execution time

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--distribution', default="gs", type=str, help='the output distribution')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#test_dataset = BSDS_RCFLoader(root='./data_file/images/1to4test', split="test")
test_dataset = BSDS_RCFLoader(root='./data_file', split="test")
test_loader = DataLoader(
    test_dataset, batch_size=1,
    num_workers=0, drop_last=True, shuffle=False)

with open('./data_file/test.lst', 'r') as f:
    test_list = f.readlines()
test_list = [split(i.rstrip())[1] for i in test_list]
assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

model_path = "./Pth/epoch-8-checkpoint.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Mymodel(args).to(device)

# 加载并修正参数
checkpoint = torch.load(model_path, map_location=device)
state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)

model.eval()

save_dir = "./test/BSDS"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
first_ten_batches = list(test_loader)[:]
start_time = time()  # Start timing
for idx, image in enumerate(first_ten_batches):
    filename = splitext(test_list[idx])[0]
    image = image.cuda()
    mean, std = model(image)
    _, _, H, W = image.shape
    outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
    outputs = torch.sigmoid(outputs_dist.rsample())
    png = torch.squeeze(1-outputs.detach()).cpu().numpy()
    result = np.zeros((H , W ))
    result[0:, 0:] = png
    result_png = Image.fromarray((result * 255).astype(np.uint8))
    png_save_dir = os.path.join(save_dir, "png")
    mat_save_dir = os.path.join(save_dir, "mat")
    os.makedirs(png_save_dir, exist_ok=True)
    os.makedirs(mat_save_dir, exist_ok=True)
    result_png.save(join(png_save_dir, "%s.png" % filename))
    io.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)
end_time = time()  # End timing

elapsed_time = end_time - start_time
print(len(test_loader))
fps = len(test_loader) / elapsed_time
print(f"Average FPS: {fps:.2f}")  # Print the average FPS


def crop_images_in_place(folder, crop_pixels=31):
    """
    裁剪文件夹中所有图片的右侧和底部各`crop_pixels`个像素，
    并用裁剪后的版本替换原始图片。
    :param folder: 包含图片的文件夹路径
    :param crop_pixels: 要从右侧和底部裁剪的像素数（默认为31）
    """
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        # 尝试将文件作为图片打开
        try:
            with Image.open(file_path) as img:
                # 获取原始尺寸
                width, height = img.size

                # 计算裁剪后的新尺寸（原始尺寸减去要裁剪的像素）
                new_width = width - crop_pixels
                new_height = height - crop_pixels

                # 检查图片是否足够大可以裁剪
                if new_width <= 0 or new_height <= 0:
                    print(f"图片 {file_name} 太小，无法裁剪 {crop_pixels} 像素")
                    continue
                # 执行裁剪操作（左，上，右，下）
                cropped_img = img.crop((0, 0, new_width, new_height))
                # 保存裁剪后的图片，替换原始文件
                cropped_img.save(file_path)
                print(f"已裁剪并替换: {file_name} (新尺寸: {new_width}x{new_height})")
        except Exception as e:
            print(f"处理 {file_name} 时出错: {e}")


# Specify your folder here
folder = r"E:\6.UAED\test\BSDS\png"

# Call the function
crop_images_in_place(folder)
