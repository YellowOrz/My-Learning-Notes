# python compare_blur_deblur.py --blur /home/xzf/Projects/STFAN/output_our_low/GoPro10211_11 --deblur /home/xzf/Projects/STFAN/datasets/our_low/test/GoPro10211_11
import argparse
import os
import cv2 as cv
import numpy as np
# 获取参数
parser = argparse.ArgumentParser(description='compare with blur and deblur image')
parser.add_argument('--blur', type=str, help='the directory of blur images')
parser.add_argument('--deblur', type=str, help='the directory of deblur images')
parser.add_argument('--gt', type=str, default=None, help='(Optional) the directory of ground truth images')
parser.add_argument('--output', type=str, default='output.avi', help='the path of output video')
args = parser.parse_args()

# 读取图像文件列表
compare = []
if not os.path.exists(args.blur):
    print('Error: Input wrong blur dir:{}'.format(args.blur))
    exit()
compare.append(os.listdir(args.blur))
if not os.path.exists(args.deblur):
    print('Error: Input wrong deblur dir:{}'.format(args.deblur))
    exit()
compare.append(os.listdir(args.deblur))
if args.gt is not None:
    if not os.path.exists(args.blur):
        print('Error: Input wrong ground truth dir:{}'.format(args.gt))
        exit()
    compare.append(os.listdir(args.gt))

# 查看图片数量是否一致
nums = [len(compare[i]) for i in range(len(compare))]
if nums != sorted(nums):
    print('Error: image nums mismatch ', nums, ' (blur deblur [gt])')
    exit()
num=nums[0]

img1 = cv.imread(os.path.join(args.blur, compare[0][0]))
img2 = cv.imread(os.path.join(args.deblur, compare[1][0]))
if args.gt is not None:
    img3 = cv.imread(os.path.join(args.gt, compare[2][0]))
    compare_img = np.hstack([img1, img2, img3])
else:
    compare_img = np.hstack([img1, img2])
video = cv.VideoWriter(args.output, cv.VideoWriter_fourcc(*'XVID'), 5,
                       compare_img.shape[:2][::-1], True)   # 图片维度为h*w*c，而视频要求输入w*h
video.write(compare_img)

# 从第1张开始循环
for i in range(num-1):
    i += 1
    img1 = cv.imread(os.path.join(args.blur, compare[0][i]))
    img2 = cv.imread(os.path.join(args.deblur, compare[1][i]))

    if img1 is None:
        print('Error: wrong image path {}'.format(args.blur, compare[0][i]))
        exit()
    if img2 is None:
        print('Error: wrong image path {}'.format(args.deblur, compare[1][i]))
        exit()
    if args.gt is not None:
        img3 = cv.imread(os.path.join(args.deblur, compare[2][i]))
        compare_img = np.hstack([img1, img2, img3])
    else:
        compare_img = np.hstack([img1, img2])

    video.write(compare_img)

video.release()