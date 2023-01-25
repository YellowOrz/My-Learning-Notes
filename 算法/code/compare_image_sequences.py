import argparse
import os
import cv2 as cv
import numpy as np

# 获取参数
parser = argparse.ArgumentParser(description='compare with several image sequence')
parser.add_argument('--frame', type=int, default=5, help='the frame of output video')
parser.add_argument('--output', type=str, default='output.avi', help='the path of output video')
parser.add_argument('--img_dirs', type=str, nargs="*", help='all directories of image sequences')
args = parser.parse_args()

# 读取图像文件列表
images_path = []
for dir in args.img_dirs:
    if os.path.exists(dir):
        print(dir, " is valid")
        images_path.append([os.path.join(dir, i) for i in sorted(os.listdir(dir))])     # os.listdir得到的列表可能是乱序的
if len(images_path) == 0:
    print("Error: no valid directory")
    exit()

# 查看图片数量是否一致
nums = [len(images_path[i]) for i in range(len(images_path))]
if nums != sorted(nums):
    print('Error: image nums mismatch ', nums)
    exit()
num=nums[0]

# 二维列表转置
images_path=[[images_path[i][j] for i in range(len(images_path))] for j in range(len(images_path[0]))]
# 读取图片对 并 拼接
images_cat = []
for pair in images_path:
    images = [cv.imread(path) for path in pair]
    images_cat.append(np.hstack(images))

# 保存视频
video = cv.VideoWriter(args.output, cv.VideoWriter_fourcc(*'XVID'), args.frame,
                       images_cat[0].shape[:2][::-1], True)   # 图片维度为h*w*c，而视频要求输入w*h
for img in images_cat:
    video.write(img)
video.release()