

[TOC]



==root身份==

# 软件

## apt安装

```shell
# 基础
apt install vim openssh-server tmux net-tools btop unzip zip rar unrar tree libssl-dev curl tldr trash-cli
# 开发
apt install git cmake cmake-curses-gui gcc g++ gdb build-essential make cloc meshlab
apt install  libpng-dev libboost-all-dev clang 
# 工具
apt install copyq flameshot xournalpp vlc wireguard-gui
snap install stretchly mission-center

# gnome
apt install gedit guake gnome-shell-extensions   

# deepin-wine
wget -O- https://deepin-wine.i-m.dev/setup.sh | sh
# sudo apt-get install com.qq.weixin.deepin
# sudo apt-get install com.qq.office.deepin

### 以下可选 ###
snap install moonlight
apt install kdenlive ncdu baobab smplayer mpv 
# 远程传输
apt install lrzsz samba smbclient cifs-utils
# win自带远程	
apt install tightvncserver xrdp		
# latex
apt install texstudio-l10n
# 录屏
# 旧版本Ubuntu：apt-add-repository ppa:maarten-baert/simplescreenrecorder
apt install simplescreenrecorder

# Razer驱动
sudo apt install software-properties-gtk
sudo add-apt-repository ppa:openrazer/stable
sudo apt update
sudo apt install openrazer-meta
sudo gpasswd -a $USER plugdev
# Razer GUI
sudo add-apt-repository ppa:polychromatic/stable
sudo apt update
sudo apt install polychromatic  # Full installation

### 舍弃 ###
# apt install indicator-cpufreq
# apt install x2goserver x2goserver-xsession x2goclient	# apt装不了的就用snap
```

| 软件名                                                       | 介绍                                                         | 软件名                                                       | 介绍                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------- |
| [guake](http://guake-project.org/)                           | [GNOME](https://www.bilibili.com/video/BV1Mx411U7cc)雷神终端（[教程](https://linux.cn/article-5507-1.html)） | [flameshot](https://flameshot.org/)                          | 截图软件                                  |
| [SimpleScreenRecorder](https://github.com/MaartenBaert/ssr)  | 录屏软件，apt安装                                            | gnome-shell-extensions                                       | GNOME的插件                               |
| [CopyQ](https://hluk.github.io/CopyQ/)                       | 剪切板管理，**强力推荐**！                                   | [tldr](https://tldr.sh/)                                     | 命令快速查询工具，**强力推荐**！          |
| [openrazer-meta](https://openrazer.github.io/#ubuntu)        | 第三方Razer驱动                                              | Kdenlive                                                     | 视频编辑软件                              |
| [polychromatic](https://github.com/polychromatic/polychromatic) | 第三方Razer GUI                                              | [deepin-wine](https://github.com/zq1997/deepin-wine)         | deepin-wine环境与应用在Ubuntu上的移植仓库 |
| [smplayer](https://www.smplayer.info/) && [mpv](https://mpv.io/) | 跨平台的视频播放器 && 逐帧播放引擎                           | [mission-center](https://gitlab.com/mission-center-devs/mission-center) | 好看的任务管理器                          |
| wireguard-gui                                                | wireguard vpn客户端的UI                   |||
| [ncdu](https://dev.yorhel.nl/ncdu)                           | 磁盘使用分析查看工具                                         | trash-cli                                                    | 命令行删除文件到回收站                    |
| btop                                                         | top、htop的替代品<br/>功能更强大                             | [MeshLab](https://snapcraft.io/meshlab)                      | 三维模型查看                              |
| [Stretchly](https://github.com/hovancik/stretchly/releases)  | 休息提醒                                                     | X2Go Client                                                  | 基于ssh的远程图形界面                     |
| cloc                                                         | 代码统计                                                     | [xournalpp](https://github.com/xournalpp/xournalpp)          | pdf编辑                                   |
| [poppler-utils](https://blog.csdn.net/Leon_Jinhai_Sun/article/details/139151611) | PDF处理工具集                                                | [pdftk](https://blog.csdn.net/weixin_43147145/article/details/104771580) | PDF处理工具集                             |
| baobab                                                       | 分析磁盘使用情况（即文件大小）                               | [moonlight](https://moonlight-stream.org/)                   | 串流客户端                                |
| [tlp](https://linrunner.de/tlp/installation/ubuntu.html)     | 电源管理工具（可降低功耗，[其他资料](https://www.reddit.com/r/linuxquestions/comments/116pc8u/lower_ryzen_frequency_and_tdp_on_linux_how_much/)） |                                                              |                                           |
| ~~[VLC](https://www.videolan.org/vlc/download-ubuntu.html)~~ | ~~视频播放器（跨平台）~~                                     | ~~indicator-cpufreq~~                                        | ~~CPU性能调节~~                           |

## 手动安装

| 软件名                                                       | 介绍                                                         | 软件名                                                       | 介绍                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------- |
| [星火商店](https://spark-app.store/) | Debian 系发行版的应用商店，**强力推荐**！                    | [Typora](https://typora.io/#download)                        | Markdown编辑器 |
| [advcpmv](https://github.com/jarun/advcpmv)                  | 让mv和cp显示进度                                             | [AppImageLauncher](https://github.com/TheAssassin/AppImageLauncher) | 集成&运行AppImage（18.04之后共用）                           |
| [OneDrive](https://github.com/abraunegg/onedrive) | 第三方onedrive软件，[20.04](https://github.com/abraunegg/onedrive/blob/master/docs/ubuntu-package-install.md#distribution-ubuntu-2004) \| [22.04](https://github.com/abraunegg/onedrive) | [Mathpix Snip](https://mathpix.com/desktop-downloads) | 数学公式识别神器            |
| [OneDriveGUI](https://github.com/bpozdena/OneDriveGUI) | 上面OneDrive的配套GUI                                        | [CLion](https://www.jetbrains.com/clion/download/#section=linux) | IDE，下载[2021.2.2版](https://download.jetbrains.com.cn/cpp/CLion-2021.2.2.tar.gz) |
| [diskusage](https://github.com/chenquan/diskusage)           | 磁盘使用情况查看                                             | [texlive](https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/Images/) | latex工具集 |
| [Ao](https://github.com/klaussinani/ao/releases)             | Microsoft To-Do desktop app                                  | [drawio](https://github.com/jgraph/drawio-desktop/releases) | 跨平台（包括网页）的流程绘制工具 |
| 搜狗输入法 |                                                              | [code-server](https://github.com/coder/code-server) | vscode的网页版 |
| ~~[福昕阅读器](https://www.foxitsoftware.cn/downloads/)~~ | pdf阅读器 |                                                              |                                                              |

## 星火商店

如果星火商店闪退，可以检查是否安装显卡驱动

| 软件名                                                       | 介绍                                                         | 软件名                                                       | 介绍                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [VSCode](https://code.visualstudio.com/download)             | linux最好用的文本显示工具                                    | [WPS](https://linux.wps.cn/)                                 | linux最好用的office工具（跨平台）                            |
| [linuxqq](https://im.qq.com/linuxqq/download.html)           | 官方qq                                                       | [百度网盘](https://pan.baidu.com/download#linux)             | 官方                                                         |
| [钉钉](https://alidocs.dingtalk.com/i/p/nb9XJlJ7QbxN8GyA/docs/ROGpvEna5YQWmaPgQ156W4ykmK3zoB27) | 官方钉钉                                                     | [clash for windows](https://github.com/Fndroid/clash_for_windows_pkg/releases) | 代理软件（跨平台）                                           |
| [网易云音乐](https://music.163.com/#/download)               | 网易云音乐（跨平台）                                         | [Mathpix Snip](https://snapcraft.io/mathpix-snipping-tool)   | 数学公式识别神器                                             |
| [UTools](https://u.tools/)                                   | 小工具集合，**强力推荐**！如果打不开参考[这个链接](https://blog.csdn.net/bugpz/article/details/124686977) | [XMind](https://www.xmind.cn/download/)                      | 思维导图                                                     |
| [腾讯会议](https://source.meeting.qq.com/download/)          | 官方                                                         | [XDM](https://xtremedownloadmanager.com/#downloads)          | 下载软件。[浏览器插件地址](https://subhra74.github.io/xdm/redirect.html?target=chrome)（跨平台） |
| [Thunderbird](https://www.thunderbird.net/zh-CN/)            | 邮箱                                                         | [zotero](https://www.zotero.org/download/)                   | 文献管理                                                     |
| [ToDesk](https://www.todesk.com/download.html)               | 远程控制软件                                                 | [向日葵](https://sunlogin.oray.com/download)                 | 远程控制软件（跨平台）                                       |
| [XnView MP](https://www.xnview.com/en/xnviewmp/#downloads)   | 图片查看软件（跨平台）                                       |                                                              |                                                              |
| [VirtualBox](https://www.virtualbox.org/wiki/Linux_Downloads) | 虚拟机                                                       | [TeamViewer](https://www.teamviewer.cn/cn/download/linux/)   | 远程控制软件                                                 |
|                                                              | python IDE                                                   | VNC Viewer                                                   | VNC远程                                                      |
| 微信(wine)                                                   | [双屏需将左屏作为主屏，否则图标显示异常](https://github.com/wszqkzqk/deepin-wine-ubuntu/issues/135#issuecomment-530186788) | [Steam](https://store.steampowered.com/about/)               | 官方                                                         |
| [Motirx](https://motrix.app/zh-CN/)                          | 下载软件。支持多种协议（包含BT、磁力）                       |                                                              |                                                              |
|                                                              |                                                              |                                                              |                                                              |
| ~~[mendeley](https://www.mendeley.com/download-desktop-new/)~~ | 论文管理工具                                                 | ~~[Free Download Manager](https://www.freedownloadmanager.org/zh/download-fdm-for-linux.htm)~~ | 下载工具                                                     |
| ~~[微信](https://blog.csdn.net/OTZ_2333/article/details/122368735)~~ | 官方微信（从优麒麟镜像安装）                                 |                                                              |                                                              |
| ~~[UEngine运行器](https://gitee.com/gfdgd-xi/uengine-runner)~~ | 运行安卓应用                                                 | ~~[Mark Text](https://github.com/marktext/marktext/releases)~~ | Markdown编辑器，开源，跨平台(似乎不维护了)                   |

## 有意思的小工具

| 软件名                                         | 介绍         | 软件名 | 介绍 |
| ---------------------------------------------- | ------------ | ------ | ---- |
| [carbonyl](https://github.com/fathyb/carbonyl) | 终端访问网页 |        |      |

# 库

| 库名称   | 说明             | 安装方法      |
| -------- | ---------------- | --------------------------------------------- |
| OpenGL | 计算机图形学库 | apt-get install libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev<br/>可选（推荐，现成demo）：apt install mesa-utils |
| Eigen    | 矩阵处理         | apt install libeigen3-dev   |
| [Pangolin](https://github.com/stevenlovegrove/Pangolin) | 可视化           | 依赖：apt install libgl1-mesa-dev libglew-dev<br>[git](https://github.com/stevenlovegrove/Pangolin)后用cmake编译安装 |
| Sophus   | 李代数           | 依赖：apt install libfmt-dev<br>[git](https://github.com/strasdat/Sophus)后用cmake编译（无需安装）  |
| [Ceres](http://ceres-solver.org/installation.html) | 求解最小二乘问题 | 依赖：apt install cmake libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev<br>[git](https://github.com/ceres-solver/ceres-solver)后用cmake编译安装 |
| g2o      | 基于图优化       | 依赖：apt install cmake libeigen3-dev libsuitesparse-dev qtdeclarative5-dev qt5-qmake qt5-default libqglviewer-dev-qt5 libcxsparse3 libcholmod3<br>[git](https://github.com/RainerKuemmerle/g2o)后用cmake编译安装 |
| [glog](https://github.com/google/glog) | 日志记录框架 | [git](https://github.com/google/glog)后用cmake编译安装 |
| OpenCV |计算机视觉库 | 见下面OpenCV专栏 |
| PCL | 点云库 | 见下面PCL专栏 |
|FLANN|最完整的（近似）最近邻开源库|见下面PCL专栏 |
| Boost | 为C++语言标准库提供扩展的一些C++程序库的总称 | 见下面PCL专栏 |
| VTK | 可视化工具库 | 见下面PCL专栏 |
| OpenNI | 开放自然交互。。。懂的都懂 | 见下面PCL专栏 |
| QHull | 计算几何库 | 见下面PCL专栏 |
| nvtop | GPU信息查看 | 依赖：apt install cmake libncurses5-dev libncursesw5-dev git<br>[git](https://github.com/Syllo/nvtop)后用cmake编译<br />也可以apt安装 |

整理成.sh如下：
```shell
# OpenGL
apt-get install libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev
# OpenGL可选
apt install mesa-utils
# 安装这些依赖的时候好像会安装python2.7，我也不知道为啥。而且安装完后运行python会自动运行python2.7，不过重新注入环境变量了以后再运行python用的就是conda里面的python，所以我也就没有管它了。
apt install gcc g++ cmake build-essential make libpng-dev libboost-all-dev -y 
apt install libeigen3-dev liblapack-dev libcxsparse3 libgflags-dev libgoogle-glog-dev libatlas-base-dev libgtest-dev cmake libsuitesparse-dev qtdeclarative5-dev qt5-qmake qt5-default libqglviewer-dev-qt5 libcxsparse3 libcholmod3 libgl1-mesa-dev libglew-dev liblz4-dev libfmt-dev -y 
# 安装Pangolin出现‘No package ‘xkbcommon’ found’
apt install libxkbcommon-x11-dev

# eigen：推荐先用apt装一下，因为很多apt装的三方库依赖apt版本的eigen
git clone -b 3.3.9 --depth=1 https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir build
cd build
cmake ..
make -j4 install
mv /usr/include/eigen3 /usr/include/eigen3_backup
ldconfig -v
cp -r /usr/local/include/eigen3 /usr/include/eigen3 
ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
cd ../..

# Pangolin
git clone -b v0.8 --depth=1 https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake ..
make -j4
make install
cd ../..

# Sophus
git clone -b 1.22.4 --depth=1 https://github.com/strasdat/Sophus.git
cd Sophus
mkdir build
cd build
cmake ..
make -j4
make install #可以不安装，但是我还是装了
cd ../..

# ceres
git clone -b 2.1.0 --depth=1 https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
mkdir build
cd build_
cmake ..
make -j4
make install
cd ../..

# g2o
git clone -b 20201223_git --depth=1 https://github.com/RainerKuemmerle/g2o.git
cd g2o
mkdir build
cd build
cmake ..
make -j4 install
cd ../..

# glog
git clone --depth=1 -b v0.6.0 https://github.com/google/glog.git
cd glog
cmake -S . -B build -G "Unix Makefiles"
cmake --build build
cmake --build build --target install
cd ..

# nvtop
git clone --depth=1 https://github.com/Syllo/nvtop.git
cd nvtop
mkdir build
cd build
cmake ..
make -j4 install
```

## PCL

- 安装依赖以及第三方库：Boost，Eigen，FlANN，VTK，（OpenNI，QHull）
   ```shell
   # 必装：其中eigen和vtk一直在更新，安装名称中的数字可能会发生变化
   apt install build-essential libboost-all-dev libeigen3-dev libvtk7-dev
   # FLANN
   git clone -b 1.9.1 --depth=1 https://github.com/flann-lib/flann.git
   cd flann
   mkdir build
   cd build
   cmake ..
   make -j7
   make install
   cd ../..
   
   # 可选2
   apt install libqhull-dev libusb-1.0-0 libusb-1.0-0-dev libopenni2-dev libopenni-dev
   ```
   PS：如果安装flann库的时候遇到下面的问题

   - cmake的时候报错`No SOURCES given to target: flann`，参考[这个链接](https://stackoverflow.com/questions/50763621/building-flann-with-cmake-fails)

   - make的时候，报错undefined reference to 'LZ4_resetStreamHC'啥的，且后面出现了matlab字样，则在cmake后面加个`-DBUILD_MATLAB_BINDINGS=OFF`

- **从[GitHub](https://github.com/PointCloudLibrary/pcl)克隆源码**

    ```shell
    git clone -b pcl-1.12.1 --depth=1 https://github.com/PointCloudLibrary/pcl.git
    cd pcl && mkdir build && cd build
    cmake ..
    # 如果想要安装Release版本，运行命令cmake -DCMAKE_BUILD_TYPE=Release ..
    
    make -j6 install
    ```



## OpenCV

### python

```shell
#只安装opencv
pip install opencv_python
#安装opencv + opencv_contrib
pip install opencv-contrib-python
```
查看可以安装的版本，在命令后面加上`==`，例如`pip install opencv_python==`

### C++

~~以下不保证最新，最新的内容可以看[我的博客](https://blog.csdn.net/OTZ_2333/article/details/104040394)~~

安装前一定先看一遍**官方教程**（[Installation in Linux](https://docs.opencv.org/4.2.0/d7/d9f/tutorial_linux_install.html)，[opencv_contrib](https://github.com/opencv/opencv_contrib)）和**以下全文**，尤其是最后的**问题**

~~以opencv 4.2.0版本为例，我`home`下的`Downloads`文件夹里有`opencv-4.2.0`、`opencv_contrib-master`和`opencv_need`三个文件夹，分别存放着opencv 4.2.0的源码、opencv contrib的源码和问题三中自己手动下载的所有文件~~  

```shell
#安装所有必须的软件和依赖项。如果显示E: Unable to locate package xxxx，把镜像源更换为清华的应该就能解决。
apt install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
#可选项。若libjasper-dev不能安装,参考问题一。除了python的两个，其他的我全装了（都是处理各种图片格式的库），libjasper-dev找不到的话就算了吧（解决方法见下面问题一）
apt install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

#获取源代码，git或者直接下载（推荐xdm）后传给linux解压
git clone -b 4.5.5 --depth=1 https://github.com/opencv/opencv.git
git clone -b 4.5.5 --depth=1 https://github.com/opencv/opencv_contrib.git
# 安装3.x版本类似如下
# git clone -b 3.4.9 --depth=1 https://github.com/opencv/opencv.git
# git clone -b 3.4.9 --depth=1 https://github.com/opencv/opencv_contrib.git

#进入opencv的文件夹
cd opencv
mkdir build
cd build

#如果报错，在-D后加个空格；
#-DOPENCV_EXTRA_MODULES_PATH=后面跟的是opencv_contrib的路径,因为我的opencv_contrib-master和opencv-4.2.0两个文件夹在同一个文件夹下
#-DBUILD_opencv_java和-DBUILD_opencv_python是用来选择是否要java和python的
# cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF ..
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF ..
#若显示	-- Configuring done
#		-- Generating done
#则进行下一步，-j7表示多线程（7为线程数）
#如果内存不是很大or编译的时候卡死，可以不开启多线程，或者把线程数减少
make -j7 
make install
```
PS：如果cmake的时候，输出说`Could NOT find xxx`之类的，不要担心，只要不是让cmake终止的`error`都没问题。cmake成功后会显示`Configuring done`和`Generating done`  

**问题一**：安装可选依赖包libjasper-dev的时候，显示`E: Unable to locate package libjasper-dev`  

```shell
add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
apt update
apt install libjasper1 libjasper-dev
add-apt-repository --remove "deb http://security.ubuntu.com/ubuntu xenial-security main"
#其中libjasper1是libjasper-dev的依赖包
```
>参考教程：[Ubuntu18.04下安装OpenCv依赖包libjasper-dev无法安装的问题](https://blog.csdn.net/weixin_41053564/article/details/81254410)  

**问题二**：如果在`cmake编译`的时候，显示**No package 'gtk+-3.0' found**  
![opencv_cmake_not_find_GTK](images/opencv_cmake_not_find_GTK.jpg)

```shell
#以防万一，我觉得还是装了比较好
apt install libgtk-3-dev
```

**问题三**：在`cmake编译`的时候，需要到Github下载一些文件，可以开启代理下载

**问题四**：报错`Duplicated modules NAMES has been found`，如下图
<img src="images/opencv_cmake_Duplicated_modules_NAMES.png" alt="opencv_cmake_Duplicated_modules_NAMES" style="zoom:67%;" />
因为*版本不匹配*！！！opencv contrib也是分版本的！！！在从github上下载opencv contrib的时候，需要选择banch。`master`指的是最新的版本，即opencv 4.x，`3.4`应该指的是opencv 3.4.x的（不懂了，那3.4.x之前的版本咋办？)，如图：
<img src="images/opencv_contrib_branch.jpg" alt="opencv_contrib_branch" style="zoom:67%;" />

 

## ARM开发板安装PoCL

- 测试环境：

    - NVIDIA JETSION ORIN NX 16GB （ubuntu20.04）
    - 九鼎创展 I3588（debian 11）

- 首先安装依赖：

    ```bash
    LLVM_VERSION=11
    sudo apt install -y build-essential ocl-icd-libopencl1 cmake git pkg-config libclang-${LLVM_VERSION}-dev clang-${LLVM_VERSION} llvm-${LLVM_VERSION} make ninja-build ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev libhwloc-dev zlib1g zlib1g-dev clinfo dialog apt-utils libxml2-dev libclang-cpp${LLVM_VERSION}-dev libclang-cpp${LLVM_VERSION} llvm-${LLVM_VERSION}-dev libncurses5
    ```

- 获取PoCL源码：[PoCL - Portable Computing Language | Download (portablecl.org)](http://portablecl.org/download.html)，我选择的是最新的4.0版本

- 编译 && 安装

    ```bash
    # 在NVIDIA开发版上，cmake命令中添加-DENABLE_CUDA=ON
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/pocl/ -DLLC_HOST_CPU=cortex-a76 -DHOST_CPU_CACHELINE_SIZE=64 ..
    make 
    sudo make install
    sudo mkdir -p /etc/OpenCL/vendors/
    sudo cp /usr/local/pocl/etc/OpenCL/vendors/pocl.icd /etc/OpenCL/vendors/
    ```

- LLC_HOST_CPU填写CPU的类型。通过命令`llc -mcpu=help`可以查看当前版本llvm支持的CPU类型
    - 如果提示`llc: command not found`，则将llc换成/usr/lib/llvm-xx/bin/llc，其中xx是llvm的版本号
    - 如果报错`error: unable to get target for 'unknown'`，则加上-march=aarch64
- NVIDIA JETSON ORIN NX的CPU型号是cortex-a78ae，但是我看llvm-11和llvm-12都只支持cortex-a78。于是我就用cortex-a78，可以编译通过

> 参考文档：
> [Build and Install OpenCL on Jetson Nano | by Muhammad Yunus | Medium](https://yunusmuhammad007.medium.com/build-and-install-opencl-on-jetson-nano-10bf4a7f0e65)
> [nvidia jetson agx xavier运行 OpenCL_jetson opencl-CSDN博客](https://blog.csdn.net/lai823177557/article/details/125386488)
> [Could not determine whether to use -march or -mcpu with clang · Issue #1170 · pocl/pocl (github.com)](https://github.com/pocl/pocl/issues/1170)

# 快捷键

## 软件

| 快捷键              | 作用                | 命令                                                  |
| ------------------- | ------------------- | ----------------------------------------------------- |
| `super + enter`     | gTile多分屏窗口管理 |                                                       |
| `super + E`         | nautilus            | /usr/bin/nautilus                                     |
| `Shift+Ctrl+Escape` | 任务管理器          | gnome-system-monitor                                  |
| `Shift+Ctrl+Alt+W`  | 打开微信            | /opt/apps/com.qq.weixin.mejituu/files/shortcut_key.sh |
| `Alt+F2`            | guake               |                                                       |
| `Super + V`         | copyq               | /usr/bin/copyq toggle                                 |
| `Shift+Print`       | flameshot截图       | /usr/bin/flameshot gui                                |



## 终端

| 快捷键     | 作用                             | 快捷键     | 作用               |
| ---------- | -------------------------------- | ---------- | ------------------ |
| `ctrl + w` | 往回删除一个单词，光标放在最末尾 | `ctrl + l` | 清屏               |
| `ctrl + u` | 删除光标以前的字符               | `ctrl + k` | 删除光标以后的字符 |
| `ctrl + a` | 移动光标至的字符头               | `ctrl + e` | 移动光标至的字符尾 |
| `ctrl + c` |                                  | `ctrl + z` |                    |
|            |                                  |            |                    |

# 桌面环境

## GNOME

- 小工具：

    ```bash
    sudo apt install gnome-sushi
    ```

- 黑暗主题

  ```bash
  # ubuntu 20.04及以上
  apt install yaru-theme-*
  # ubuntu 18.04
  snap install communitheme
  ```

- **<u>gnome插件</u>**：`apt install gnome-shell-extensions`

  - 查看gnome版本：系统设置=>关于=>GNOME版本

  - 使用教程：[通过浏览器插件安装](https://linux.cn/article-9447-1.html)，[手动安装](https://www.debugpoint.com/manual-installation-gnome-extension/)

      - 如果插件网页显示“你的本地主机连接器不支持下列 API：v6。或许你可以升级本地主机连接器，或者安装用于缺失 API 的插件”，不用管，继续安装插件就好

  - **[cpupower](https://github.com/deinstapel/cpupower)** ：CPU性能调节插件

      ```bash
      sudo add-apt-repository ppa:fin1ger/cpupower
      sudo apt-get update
      sudo apt-get install gnome-shell-extension-cpupower
      # 重新登录后
      gnome-extensions enable cpupower@mko-sl.de
      ```

  - ~~[GNOME顶部栏的图标隐藏](https://blog.csdn.net/hwh295/article/details/113733884)：[Icon Hider](https://extensions.gnome.org/extension/351/icon-hider/)~~

  - ~~[时间挪到右上角](https://blog.csdn.net/ricolau/article/details/120853288)：[Frippery Move Clock](https://extensions.gnome.org/extension/2/move-clock/)~~

  - [定制GNOME](https://www.reddit.com/r/gnome/comments/pb5y81/how_to_customize_the_top_bar_in_gnome/)：[Just Perfection](https://extensions.gnome.org/extension/3843/just-perfection/)

      - ☆**修改顶栏时间位置**：找到Just Perfection的设置，定制=>时钟菜单位置=>右边

  - **多分屏窗口管理**：[gTile](https://extensions.gnome.org/extension/28/gtile/)，效果如下（默认快捷键为`super键`+`enter`。类似的还有[Put Windows](https://extensions.gnome.org/extension/39/put-windows/)

      ![img](images/screenshot_28.png)

- 性能指示器：[indicator-sysmonitor](https://github.com/fossfreedom/indicator-sysmonitor)  ，显示于状态栏，支持GNOME和MATE，程序里面设置**开机自启**，显示格式为

  ```
  {netcomp}║C:{cpu}║G:{nvgpu}║M:{mem}
  ```

- 快捷键设置：Settings=>Keyboard=>Keyboard Shortcuts

  | 名称          | 命令                             | 快捷键               | 系统默认                 |
  | ------------- | -------------------------------- | -------------------- | ------------------------ |
  | flameshot截图 | /usr/bin/flameshot gui           | `shift + prtsc`      | 截图=>截图               |
  | 任务管理器    | htop                             | `ctrl + shift + esc` |                          |
  | nautilus      | /usr/bin/nautilus                | `windows + e`        |                          |
  | copyq主界面   | /usr/bin/copyq toggle            | `windows + v`        | 系统=>显示通知列表       |
  | Guake         | /usr/bin/guake                   | `Alt + F2`           | 系统=>显示运行命令提示符 |
  | utools        | 不需要添加，把系统默认的禁止即可 | `Alt + Space`        | 窗口=>激活窗口菜单       |

- `gnome`开启屏幕共享：

  <img src="images/image-20220908101428087.png" alt="image-20220908101428087" style="zoom: 67%;" />

  若共享下面没有“屏幕共享选项”，则安装

  ```bash
  sudo apt install vino
  ```

  若使用客户端连接时报错类似“No security types sypported”或者“vnc连接提示不支持安全类型”，则打开`dconf-editor`（可以用`apt`安装），找到`org=>gnome=>desktop=>remote-access=>require-encryptyion`，关闭即可

- alt+tab切换窗口而非应用（即不要将同应用的不同窗口折叠）：apt安装`dconf-editor`，进入`org/gnome/desktop/wm/keybindings`下，把`switch-applications`中的`'\<Alt\>Tab'`删去，在`switch-windows`（或者`switch-group`）中加入`'\<Alt\>Tab'`

- 管理**开机自启**：打开gnome-tweaks（可以用apt安装，中文可能叫“优化”），在开机启动程序添加如下软件

    - 软件内设置：clash for windows、
    - 手动添加：xdm、stretchly、utools、Guake、flameshot、thunderbird、钉钉、indicator-sysmonitor

- 设置nautilus自带终端：参考[nautilus-terminal](https://github.com/flozz/nautilus-terminal)，注意必须要用apt安装的python，不能用conda安装的python

    <img src="images/image-20220916141444464.png" alt="image-20220916141444464" style="zoom:67%;" />

## XFCE

- [安装xrdp](https://www.golinuxcloud.com/install-xrdp-with-xfce4-on-ubuntu/)：

    ```bash
    sudo apt install xrdp -y
    sudo systemctl enable xrdp
    sudo systemctl start xrdp
    echo xfce4-session >> ~/.xsession
    ```

    编辑`/etc/xrdp/startwm.sh`，在`test -x /etc/X11/Xsession && exec /etc/X11/Xsession`之前加入内容

    ```bash
    unset DBUS_SESSION_BUS_ADDRESS
    unset XDG_RUNTIME_DIR
    ```

    最后重启服务

    ```bash
    sudo systemctl restart xrdp
    ```

- 开机自启：设置 => Session and Startup => Application Autostart

- 快捷键设置：设置=>Keyboard=>Application Shortcuts

    | 名称          | 命令                             | 快捷键               | 系统默认功能                  |
    | ------------- | -------------------------------- | -------------------- | ----------------------------- |
    | flameshot截图 | /usr/bin/flameshot gui           | `shift + prtsc`      | xfce4-screenshooter -r        |
    | 任务管理器    | /snap/bin/mission-center         | `ctrl + shift + esc` | xfce4-taskmanager             |
    | nautilus      | /usr/bin/thunar                  | `windows + e`        | exo-open --launch FileManager |
    | copyq主界面   | /usr/bin/copyq toggle            | `windows + v`        |                               |
    | 下拉终端      | xfce4-terminal --drop-down       | `Alt + F2`           | xfrun4                        |
    | utools        | 不需要添加，把系统默认的禁止即可 | `Alt + Space`        | Window operations menu        |

- 系统代理：编辑`/etc/environment`

    ```bash
    no_proxy=localhost,127.0.0.0/8,*.local,192.168.0.0/16
    http_proxy=http://192.168.1.2:7890
    https_proxy=http://192.168.1.2:7890
    NO_PROXY=localhost,127.0.0.0/8,*.local,192.168.0.0/16
    HTTP_PROXY=http://192.168.1.2:7890
    HTTPS_PROXY=http://192.168.1.2:7890
    ```

- 设置锁屏时间：设置=>Xfce Screensaver=>Lock Screen=>Lock Screen with Screeensaver

# 通用配置

- [fcitx5安装rime](https://www.cnblogs.com/Undefined443/p/-/rime)：

    ```bash
    # 安装 Fcitx5 框架和中文插件
    sudo apt install fcitx5 fcitx5-chinese-addons
    # 安装基于 Fcitx5 的 RIME 引擎
    sudo apt install fcitx5-rime librime-plugin-lua
    
    cd /opt
    curl -fsSL https://raw.githubusercontent.com/rime/plum/master/rime-install | sudo bash
    sudo chmod a+w -R plum	
    rime_frontend=fcitx5-rime bash ./plum/rime-install double-pinyin
    ```

    - 配置：编辑`~/.local/share/fcitx5/rime/default.custom.yaml`，添加如下内容

      ```bash
      patch:
        # 要启用的输入方案
        schema_list:
          - schema: double_pinyin_flypy   # 小鹤双拼
        menu:
          # 候选词个数
          page_size: 4
      ```
      
    - [切换简体中文](https://miaostay.com/2018/11/rime%E8%AE%BE%E7%BD%AE%E4%B8%BA%E9%BB%98%E8%AE%A4%E7%AE%80%E4%BD%93/)：编辑`~/.local/share/fcitx5/rime/double_pinyin_flypy.schema.yaml`

      ```bash
      # 将如下内容
        - name: simplification
          states: [ 漢字, 汉字 ]
      # 改成
        - name: simplification
          reset: 1
          states: [ 漢字, 汉字 ]
      ```
      

- root登录：[Ubuntu 18.04-20.04开机自动root用户登录（测试可用）_ubuntu开机进入root用户_墨痕诉清风的博客-CSDN博客](https://blog.csdn.net/u012206617/article/details/122343463)

- 更改镜像为[清华镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)or[阿里镜像源](https://developer.aliyun.com/mirror/ubuntu?spm=a2c6h.13651102.0.0.3e221b119vwOjw)

- **[同步双系统时间](https://zhuanlan.zhihu.com/p/492885761)**：终端运行命令`timedatectl set-local-rtc 1`，然后到windows下面再更新一下时间

- [双网卡设置流量](https://blog.csdn.net/cuiyuan605/article/details/113397856)：

    - 假设网卡信息和原始的路由信息如下：

        |      | 网卡1        | 网卡2       |
        | ---- | ------------ | ----------- |
        | 名称 | enp37s0      | wlp38s0     |
        | IP   | 10.168.2.2   | 192.168.2.3 |
        | 网关 | 10.168.2.254 | 192.168.2.1 |
        | 掩码 | 23           | 24          |

        ```bash
        orz@orz:~$ ip route show
        default via 10.168.2.254 dev enp37s0 proto dhcp metric 20100 
        default via 192.168.2.1 dev wlp38s0 proto dhcp metric 20600 
        10.168.1.0/23 dev enp37s0 proto kernel scope link src 10.168.2.2 metric 100 
        169.254.0.0/16 dev enp37s0 scope link metric 1000 
        192.168.2.0/24 dev wlp38s0 proto kernel scope link src 192.168.2.3 metric 600
        ```

    - 设置优先网卡：即将跃点改小即可，比如将网卡2设置优先，命令如下

        ```bash
        ip route del default via 192.168.2.1 dev wlp38s0 proto dhcp metric 20600
        ip route add default via 192.168.2.1 dev wlp38s0 proto dhcp metric 1
        ```

    - 指定网段orIP走指定网卡：比如都走网卡1

        ```bash
        # 指定局域网网段
        ip route add 10.168.1.0/24 dev enp37s0 proto kernel scope link src 10.168.2.2 metric 100
        # 指定局域网IP
        ip route add 10.168.1.3 dev enp37s0 proto kernel scope link src 10.168.2.2 metric 100
        # 指定公网IP
        ip route add 117.21.179.19 via 10.168.2.254 dev enp37s0 proto dhcp metric 10
        ```
        

- [双拼输入法](https://blog.nickwhyy.top/post/ubuntudoublepin/)：小鹤双拼

    ```bash
    sudo apt install ibus-rime librime-data-double-pinyin
    mkdir -p ~/.config/ibus/rime
    ```

    编辑文件`~/.config/ibus/rime/default.custom.yaml`，内容如下

    ```bash
    patch:
        schema_list:
            - schema: double_pinyin_flypy # 小鹤双拼  
        menu: 
            page_size: 4 				# 设置后选词的个数
    ```

    在`~/.bashrc`添加如下内容

    ```bash
    export GTK_IM_MODULE=ibus
    export XMODIFIERS=@im=ibus
    export QT_IM_MODULE=ibus
    ```

- 美化Bash：

    - 方法一：编辑`~/.bashrc`，取消`force_color_prompt=yes`前的注释

    - 方法二：在任意一个环境变量文件（比如`/etc/bash.bashrc`or `/etc/bashrc`，如果里面已经有一个PS1了，可以注释掉）添加如下代码，添加完后重新注入环境变量

        ```bash
        PS1="\[\e[36;1m\]\u\[\e[0m\]@\[\e[33;1m\]\h\[\e[0m\]:\[\e[31;1m\]\w\[\e[0m\]\$ "
        PS1='\[\033[0;31m\]\342\224\214\342\224\200$([[ $? != 0 ]] && echo "[\[\033[1;31m\]\342\234\227\[\033[0;31m\]]\342\224\200")[\[\033[0;9m\]\u@\h\[\033[0;31m\]]\342\224\200[\[\033[0;32m\]\w\[\033[0;31m\]]\n\[\033[0;31m\]\342\224\224\342\224\200\342\224\200\342\225\274 \[\033[0m\]\[\e[01;33m\]\$\[\e[0m\] '
        ```

        > [最好看的Bash美化——打造ParrotOS风格的Bash](https://blog.csdn.net/u011145574/article/details/105160496)


- Tmux：在配置文件`~/.tmux.conf `中加入如下内容，然后重启tmux，或者按`ctrl+b`后输入`:source-file ~/.tmux.conf`

    ```shell
    # 启用鼠标
    set -g mouse on
    # set-option -g mouse on # 或者这个
    # 复制模式
    set-window-option -g mode-keys vi #可以设置为vi或emacs
    # 开启256 colors支持
    set -g default-terminal "screen-256color"
    # set-window-option -g utf8 on #开启窗口的UTF-8支持，报错
    # 多窗口同步操作快捷键ctrl+b s
    bind-key s setw synchronize-panes
    set -g history -limit 99999
    ```

    复制模式步骤：

    1. `ctrl+b`，然后按`[`进入复制模式
    2. 移动鼠标到要复制的区域，移动鼠标时可用vim的搜索功能"/","?"
    3. 按空格键开始选择复制区域
    4. 选择完成后按`enter`退出，完成复制
    5. `ctrl+b` ，然后按`]`粘贴

- 终端根据历史补全命令：编辑`/etc/inputrc`，搜索关键字`history-search`找到如下两行，取消注释。保存退出后即可通过`PgUp`和`PgDn`根据历史补全命令

    <img src="images/image-20200923101318000.png" alt="image-20200923101318000" style="zoom:90%;" />

- 给应用添加代理：基于electron的软件，例如microsoft-edge、AO等

    - 编辑文件`/usr/share/applications/*.desktop`（或者在路径`~/.local/share/applications/`下），在`Exec=`后面的内容中加上`--proxy-server="http://127.0.0.1:7890"`，例如
    ```bash
    # before
    Exec=/usr/bin/microsoft-edge-stable %U
    # after
    Exec=/usr/bin/microsoft-edge-stable --proxy-server="http://127.0.0.1:7890" %U
    ```

- deepin-wine安装windows软件：
    ```bash
    # .msi installer
    deepin-wine6-stable msiexec /i $PATH_TO_MSI_INSTALL
    ```

- [设置桌面为默认.descktop路径](https://unix.stackexchange.com/questions/391915/where-is-the-path-to-the-current-users-desktop-directory-stored)

    ```bash
    xdg-user-dir DESKTOP
    ```

- ssh代理：只支持sock5代理，在`~/.ssh/config`中添加如下内容，7891为sock5端口

    ```bash
    ProxyCommand nc -X 5 -x 127.0.0.1:7891 %h %p
    ```

- linux版本的钉钉和腾讯会议 语音时如果别人讲话有**杂音**，执行下面命令

    ```bash
    sudo apt install pulseaudio*
    ```

    > 参考[链接](https://bbs.archlinuxcn.org/viewtopic.php?id=12535)

- 英伟达显卡驱动安装：[我的博客](https://blog.csdn.net/OTZ_2333/article/details/108604064)

- 

- [swap扩容](https://blog.csdn.net/wdwangye/article/details/109371782)

    ```bash
    dd if=/dev/zero of=/swapfile bs=2G count=8
    mkswap /swapfile
    chmod 0600 /swapfile
    swapon /swapfile
    vim /etc/fstab
    # 加入如下一行
    /swapfile                                 none            swap    sw              0       0
    ```

- [修改键盘小红点灵敏度](https://blog.csdn.net/weixin_36242811/article/details/88808015)：

    ```shell
    id=$(xinput --list |grep TrackPoint| awk '{printf $6"\n"}'|cut -d "=" -f 2)
    num=$(xinput list-props $id| grep Speed|head -1|awk '{printf $4 "\n"}'| cut -d"(" -f2|cut -d")" -f1)
    xinput set-prop $id $num -1.0 
    ```

- 笔记本触控板多指功能失效：安装fusuma（[18.04](https://hirosht.medium.com/gestures-on-ubuntu-18-04-xorg-2fe05efb05fc)）

- **关闭fcitx的中文简体繁体切换快捷键：**

    <img src="images/image-20221209145039632.png" alt="image-20221209145039632" style="zoom: 80%;" />

- 字体配置：

    - 命令行界面的中文字体：TODO

    - [使用win字体](https://zhuanlan.zhihu.com/p/109083570)：

        ```bash
        cd /usr/share/fonts
        mkdir win_fonts
        chmod 755 win_fonts
        # 进入win字体所在路径（C:/Windows/Fonts/）
        cp *.ttf /usr/share/fonts/win_fonts/
        cp *.TTF /usr/share/fonts/win_fonts/
        cp *.otf /usr/share/fonts/win_fonts/
        cp simsun.ttc /usr/share/fonts/win_fonts/
        cd /usr/share/fonts/win_fonts/
        chmod 644 *
        mkfontscale
        mkfontdir
        fc-cache          #更新字体缓存
        ```

    - WPS缺少的字体可以去[这里](http://xiazaiziti.com/)下载，然后参考上面的方法添加到系统。

        - 本文档同目录下`./Material/字体/`有3种下载好的字体，请注意使用范围

- 添加代理：在环境变量（最好是`/etc/bash.bashrc` or `/etc/bashrc`）中添加如下内容

    ```shell
    # 在WSL为下面那个
    # IP=$(cat /etc/resolv.conf |grep name|cut -f 2 -d " ") 
    IP=127.0.0.1
    Port=7890
    proxyon(){
    	export http_proxy="http://${IP}:${Port}"
    	export https_proxy="http://${IP}:${Port}"
    	#    export http_proxy="socks5://${IP}:${Port}"
    	#    export https_proxy="socks5://${IP}:${Port}"
    	echo "proxy on, and IP is $(curl ip.sb)"
    }
    proxyoff(){
      unset http_proxy
      unset https_proxy
      echo "proxy off"
    }
    ```

    注意：git就不要添加类似这种的代理了，使用http代理或者ssh的代理

- utools搜索硬盘文件：https://github.com/shenzhuoyan/utools_linux_find

- SMPlayer设置：

  - 使用快捷键`.`和`,`逐帧前进、后退：“首选项”=>"常规"=>"常规"=>"多媒体引擎"，改成mpv。如果提示找不到，则安装`sudo apt install mpv`
  - 播放暂停切换窗口黑屏的解决方法：“首选项”=>"高级"=>"高级"=>"重绘视频窗口背景"，取消勾选
  - 音量调节：“首选项”=>"键盘和鼠标"=>"键盘"，找到"音量-"和"音量+"，分别改成↑和↓

  > 参考：[Ubuntu下SMPlayer播放器安装配置以及常用快捷键记录](https://blog.csdn.net/zhanghm1995/article/details/109408954)

- [code-server使用官方插件](https://blog.csdn.net/qq_45689158/article/details/125461308)：在文件`/usr/lib/code-server/lib/vscode/product.json`中最大的花括号内，新增如下内容（记得在前面一行的后面加个逗号）

  ```json
  "extensionsGallery": {
      "serviceUrl": "https://marketplace.visualstudio.com/_apis/public/gallery",
      "cacheUrl": "https://vscode.blob.core.windows.net/gallery/index",
      "itemUrl": "https://marketplace.visualstudio.com/items",
      "controlUrl": "",
      "recommendationsUrl": ""
  }
  ```

- code-server插件存放位置：`~/.local/share/code-server/extensions/`

# 技巧

- [查看swap占用情况](https://blog.csdn.net/qq_21959403/article/details/117663424)：前10个

  ```shell
  for i in $( cd /proc;ls |grep "^[0-9]"|awk ' $0 >100') ;do awk '/Swap:/{a=a+$2}END{print '"$i"',a/1024"M"}' /proc/$i/smaps 2>/dev/null ; done | sort -k2nr | head -10
  ```

  

# ~~WSL~~

- [wsl自动使用win的代理](https://github.com/microsoft/WSL/issues/10753): 编辑文件`%USERPROFILE%\.wslconfig`

    ```c++
    [experimental]
    autoMemoryReclaim=gradual  # gradual  | dropcache | disabled
    networkingMode=mirrored
    dnsTunneling=true
    firewall=true
    autoProxy=true
    ```

- **安装**：现在BIOS打开虚化，然后控制面板 -> 启用或关闭Windows功能 中，打开虚拟机平台、适用于Linux的Windows子系统、Hyper-V，重启。

    可以到微软商店直接安装。或者到[这里](https://docs.microsoft.com/zh-cn/windows/wsl/install-manual#installing-your-distro)安装，（如果是用IDM下载，格式可能会变为.zip，需要改为.appx），然后在PowerShell运行

    ```powershell
    Add-AppxPackage .\Ubuntu_1804.2019.522.0_x64.appx
    ```

    **PS**：如果想要安装到其他分区，在[这里](https://docs.microsoft.com/zh-cn/windows/wsl/install-manual#downloading-distributions)下载安装包，拷贝到其他分区想要存放的地方，将文件后缀的.appx改成.zip，然后解压，然后运行ubuntu.exe即可

- **WSL1升级为WSL2**：参考[教程](https://www.liumingye.cn/archives/326.html)or官网

- **与CLion连接**：参考CLion的官方教程[WSL | CLion (jetbrains.com)](https://www.jetbrains.com/help/clion/how-to-use-wsl-development-environment-in-product.html)

- **图形界面**：在WLS安装图像界面，例如xfce4

    ```shell
    apt install xfce4
    ```

    然后在`~/.bashrc`中添加如下[内容](https://zhuanlan.zhihu.com/p/151853503)

    ```shell
    export DISPLAY=`cat /etc/resolv.conf | grep nameserver | awk '{print $2}'`:0.0
    # 或者这个？
    # export DISPLAY=localhost:0.0
    ```

    Windows安装[MobaXTerm](https://mobaxterm.mobatek.net/download.html)。然后运行MobaXTerm，保证其X server为开启状态，即左上角的“X”为彩色，为灰色的话，按一下就彩色了

    ![MobaXTerm_X](images/MobaXTerm_X.png)

    在WSL上运行如下命令就会出现图形界面了

    ```bash
    startxfce4
    ```

    如果只是想查看运行结果（比如OpenCV的imshow），可以不执行`startxfce4`，直接执行代码就会自动打开窗口。

    **注意：**环境变量DISPLAY中的`0.0`需与MobaXTerm中的X11设置保持一致

    <img src="images/MobaXTerm_offset.png" alt="MobaXTerm_offset" style="zoom: 80%;" />

    或者MobaXterm可能会自动识别到WSL的图形界面，双击打开就好了

    ![image-20201002190701696](images/image-20201002190701696.png)

- **安装NVIDIA GPU驱动**：将win10升级至版本≥21H2 或者使用win11，然后给win[安装NVIDIA驱动](https://developer.nvidia.com/cuda/wsl)，不需要给WSL安装驱动。具体参考[微软官方教程](https://docs.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)或者更加详细的[NIVIDIA官方教程官网](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

- `updatedb`排除文件夹`/mnt`：

    - 暂时排除：使用参数`-e`

        ```shell
        updatedb -e /mnt
        ```

    - 永久排除：编辑文件`/etc/updatedb.conf`，在变量**PRUNEPATHS**里的后面加上`/mnt`，例如

    <img src="images/image-20201018111355994.png" alt="image-20201018111355994" style="zoom:80%;" />

- **监控远程服务器的GPU**：在win中安装软件**[ wsl-notify-send](https://github.com/stuartleeks/wsl-notify-send)**，然后使用如下脚本。显存占用量低于{显存大小}的GPU会被认定为空，然后在windows中发送通知

    ```shell
    ./tool/Material/monitor_GPU.sh {服务器代号} {显存大小}
    ```

    ![image-20211103134943564](images/image-20211103134943564.png)

