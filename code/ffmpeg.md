# 基础知识

- **编码格式**：视频和音频都需要经过编码，才能保存成文件。不同的编码格式（CODEC），有不同的压缩率，会导致文件大小和清晰度的差异。

  常用的**视频编码格式**有：（有版权但免费使用的三个）H.262、H.264、H.265、（无版权）VP8、VP9、AV1

  常用的**音频编码格式**有：MP3、AAC

- **编码器（encoders）**：实现某种编码格式的库文件。只有安装了某种格式的编码器，才能实现该格式视频/音频的编码和解码。

  使用命令`ffmpeg -encoders`可以查看ffmpeg已经安装的编码器

  H.264 编码器有libx264（据说是最流行&开源）、NVENC（NVIDIA）、libopenh264（也是开源的）

  使用apt安装的ffmpeg的H.264 编码器只有libopenh264，如果想用其他的编码器，ffmpeg必须是从源码安装的，并且编译的时候开启了libx264之类的编码器

# 参数

## 格式

```shell
ffmpeg [全局参数] [输入文件参数] -i [输入文件] [输出文件参数] [输出文件]
```

## 基础

- `-c`：指定编码器
  - `-c copy`：直接复制，不经过重新编码（这样比较快）
  - `-c:v`：指定视频编码器，例如`-c:v libx264`、`-c:v libopenh264`。[等价于](https://ffmpeg.org/pipermail/ffmpeg-user/2017-February/035335.html)`-codec:video`、`code:v`、`vcodec`
  - `-c:a`：指定音频编码器
- `-i`：指定输入文件。视频、图片、音频等
- `-an`：去除音频流。a表示视频，n表示删除
- `-vn`： 去除视频流。v表示音频，n表示删除
- `-vf`：
- `-preset`：（libx264专属）指定输出的视频质量（压缩比率），会影响文件的生成速度，有以下几个可用的值 ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow。
- `-y`：不经过确认，输出时直接覆盖同名文件。
- `-hide_banner`：隐藏不必要的多余讯息
- `-crf`：指定视频的品质，默认值为 23，取值范围为 0-51，值越小品质越高，0 为无损，取值为 18 时几乎是肉眼无损压制。

> [FFmpeg视频转码技巧之-crf参数（H.264篇）](https://blog.csdn.net/happydeer/article/details/52610060)
>
> [FFmpeg 视频处理入门教程](https://www.ruanyifeng.com/blog/2020/01/ffmpeg.html)

# 技巧

- 将视频转为图片序列

  ```shell
  ffmpeg -i test.mp4 %03d.jpg
  ```

- 将图片序列转为视频

  ```shell
  ffmpeg [-r {帧率}] -i %05d.png [-c:v libx265] [-b:v 300000k] test.mp4
  ```

  > 注意：
  >
  > 1. 参数`-r`不能放在`-i`后面，否则不起作用
  > 2. 如果视频要插入PPT，需要使用编码器libx265，否则插入的时候会卡
  
  如果要**[无损转换](https://qastack.cn/video/7903/how-to-losslessly-encode-a-jpg-image-sequence-to-a-video-in-ffmpeg)**
  
  ```shell
  ffmpeg [-framerate {帧率}] -i %05d.png -codec copy test.mkv
  # .mkv 指定帧率不能使用-r参数
  ```
  
  然后可以比较每个帧的哈希值，以确保结果是真正的无损
  
  ```shell
  ffmpeg -i %05d.png -f framehash -
  #stream_index, packet_dts, packet_pts, packet_duration, packet_size, hash
  #0,          0,          0,        1,   460800, 29bcc2db3726c7dfec1826c5740f603f
  #0,          1,          1,        1,   460800, b5fdc23d93cbd043dc2b9290dc8378f0
  #0,          2,          2,        1,   460800, ee0709942f24b458fd2380d134dcb59d
  ffmpeg -i test.mkv -map 0:v -f framehash -
  #stream_index, packet_dts, packet_pts, packet_duration, packet_size, hash
  #0,          0,          0,        1,   460800, 29bcc2db3726c7dfec1826c5740f603f
  #0,          1,          1,        1,   460800, b5fdc23d93cbd043dc2b9290dc8378f0
  #0,          2,          2,        1,   460800, ee0709942f24b458fd2380d134dcb59d
  ```
  
  