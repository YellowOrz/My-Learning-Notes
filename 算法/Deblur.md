# 工具

## 图像序列对比

可将（任意）多个图像序列（比如blur、deblur、gt）放在一个视频中进行对比。主体代码见[compare_image_sequences.py](./code/compare_image_sequences.py)。

- 单个场景对比的使用方法

  ```shell
  python compare_image_sequences.py [ --frame 帧率 --output 输出视频路径(格式为.avi) ] --img_dirs 序列1路径 序列2路径 ...
  ```

- 多个场景对比的使用方法

  ```shell
  dir="/home/xzf/Projects/STFAN/datasets/our_low/test/"
  for video in $(ls -d ~/Projects/STFAN/datasets/our_low/test/*/ |rev | cut -d '/' -f 2|rev )
  do
      py="/home/xzf/Projects/compare_blur_deblur_gt.py"
      blur_dir="$dir$video/input"
      STFAN_dir="/home/xzf/Projects/STFAN/output_our_low/$video/"
      CDVD_dir="/home/xzf/Projects/STFAN/output_our_low/$video/"
      output="/home/xzf/Projects/STFAN/output_our_low/$video.avi"
      echo python $py --output $output --img_dirs $blur_dir $STFAN_dir $CDVD_dir
      python $py --output $output --img_dirs $blur_dir $STFAN_dir $CDVD_dir
  done 
  ```

  