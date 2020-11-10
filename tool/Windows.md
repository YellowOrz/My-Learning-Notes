[TOC]

# 系统设置

- 分区：系统+主要的软件装在SSD上面。
    - 若只有SSD：C盘至少150G，剩下所有的容量都分给D盘
    - SSD+HDD：SSD全分给C盘，HDD全分给其他盘
- OneDrive：将OneDrive文件夹存放在C盘以外的地方，然后桌面文件夹放在OneDrive下面
- 开启剪贴板历史记录：按`Windows 键` + `V`
- 开机启动管理：任务管理器 -> 启动

# 必装软件

+ 必装软件分三种：
    + 通过**软件包安装**而且比较平常的（比如TIM、微信等）
    + 通过**压缩包解压**的（比如PdgCntEditor、OfficeBox）
    + **库软件**（比如OpenCV，Anaconda）：==如果可以全部装在WSL里面==
+ 所有的软件都**安装在C盘**，软件包安装的放在`Program Files`或者`Program Files (x86)`文件夹，压缩包解压的放在`Soft`文件夹，库软件放在`Library`文件夹。库文件的源码可以放在其他地方，如果可以的话不要删掉。
## 软件包安装
1. 不做介绍的软件：[TIM](https://tim.qq.com/download.html)、[微信](https://pc.weixin.qq.com/)、[网易云音乐](https://music.163.com/download)、[网易有道词典](https://cidian.youdao.com/index.html)、[钉钉](https://www.dingtalk.com/download)、[福昕阅读器](https://www.foxitsoftware.cn/downloads/PDF)、[阿里旺旺](https://wangwang.taobao.com/)、[百度网盘](https://pan.baidu.com/download)、[GitHub Desktop](https://help.github.com/cn/desktop/getting-started-with-github-desktop/installing-github-desktop)、[OneDrive](https://www.microsoft.com/zh-cn/microsoft-365/onedrive/online-cloud-storage)、[VSCode](https://code.visualstudio.com/download)、[Bandizip](https://cn.bandisoft.com/bandizip/)、[Edge](https://www.microsoft.com/zh-cn/edge/)、[PotPlayer](https://daumpotplayer.com/download/)、[IDM](./Material/IDM.6.38.Build.2.Final.Retail.zip) 、[按键精灵](http://download.myanjian.com/)、[Mendeley](https://www.mendeley.com/download-desktop/)、[向日葵](https://sunlogin.oray.com/personal/download/)、[Foxmail](https://www.foxmail.com/)、[Everything](https://www.voidtools.com/zh-cn/downloads/)、[欧路词典](https://www.eudic.net/v4/en/app/download)
2. 稍作介绍的软件：

| 软件名 | 介绍                | 软件名      | 介绍|
| :---: | :---: | :---: | :---: |
| ~~[XShell+Xftp](https://www.netsarang.com/zh/free-for-home-school/)~~ | ~~远程连接linux的好工具~~ | [X2Go](https://wiki.x2go.org/doku.php/download:start) | 远程连接Linux的图形界面 |
|[flux](https://justgetflux.com/)|调节屏幕色温|[stretchly](https://github.com/hovancik/stretchly/releases)|休息提醒|
|[MobaXTerm](https://mobaxterm.mobatek.net/download.html)|远程连接linux的好工具|[Typora](https://typora.io/#download)|Markdown编辑器|
|[cloc](https://github.com/AlDanial/cloc)|代码行数统计|[AIDA64](https://www.aida64.com/downloads)|显示电脑详细信息|
|  [Mathpix Snip](https://mathpix.com/#downloads)  |     数学公式识别神器      |[clash for windows](https://github.com/Fndroid/clash_for_windows_pkg/releases)|ShadowsocksR的替代品。更好用。[教程](https://merlinblog.xyz/wiki/cfw.html)，[汉化补丁](https://github.com/BoyceLig/Clash_Chinese_Patch/releases)，[ssr订阅链接转clash](https://bianyuan.xyz/)<br>但是好像会占用很高，而且影响连接|
|~~[唧唧Down](http://client.jijidown.com/)~~ | ~~下载b站视频~~ | PdgCntEditor | 编辑PDF目录                        |
| [DiskGenius](https://www.diskgenius.cn/download.php)  | 管理磁盘     | [Scrcpy_GUI](https://github.com/Tomotoes/scrcpy-gui/releases) | 手机投屏到电脑（可以媲美华为的投屏） |
| [哔哩哔哩动画](https://www.microsoft.com/zh-cn/p/%e5%93%94%e5%93%a9%e5%93%94%e5%93%a9%e5%8a%a8%e7%94%bb/9nblggh5q5fv?activetab=pivot:overviewtab) | 官方渠道下载b站视频 | [ShadowsocksR](https://github.com/shadowsocksrr/shadowsocksr-csharp) | 必备工具                       |
| [万彩办公大师](http://www.wofficebox.com/) | 办公工具~~箱~~ | ~~[SpeedPan](http://supanx.com/speedpan-free.html)~~ | ~~百度网盘资源下载器，免费版，用来搜资源~~ |
| ~~[软媒魔方](https://mofang.ruanmei.com/)~~ | ~~管理电脑的工具箱~~ | [SpeedPanX](http://supanx.com/)    | 百度网盘资源下载器，付费版，用来下资源 |
|[每日英语听力](https://www.eudic.net/v4/en/app/ting)|闭眼休息的时候听听英语|[spacedesk](https://spacedesk.net/##box_434)|将手机or平板变成电脑显示器|

3. 大型软件：[Lightroom Classic](https://mp.weixin.qq.com/s/RH0oCJWD00QFXpuCYwB8oA)、[Photoshop](https://mp.weixin.qq.com/s/RH0oCJWD00QFXpuCYwB8oA)、[Microsoft Office](https://mp.weixin.qq.com/s/RH0oCJWD00QFXpuCYwB8oA)、[WLS（Ubuntu 18.04 LTS）](https://docs.microsoft.com/zh-cn/windows/wsl/install-win10)、[Visio](https://mp.weixin.qq.com/s/RH0oCJWD00QFXpuCYwB8oA)、[CLion](https://www.jetbrains.com/clion/download/#section=windows)、[SolidWorks](https://mp.weixin.qq.com/s/RH0oCJWD00QFXpuCYwB8oA)

## 库软件

==如果可以，全部装在WSL里面==；否则全部安装于`C:\Library\`中。源码安装方法参考[CrossPlatform的文档](./CrossPlatform.md)

| 软件名 | 介绍| 软件名| 介绍|
| ---------- | ------ | ------------ | ------ |
| [Miniconda3](https://www.anaconda.com/distribution/) | 管理python| ~~CTEX~~ | ~~LaTeX发行版，毕设用~~ |
| [mingw64](https://sourceforge.net/projects/mingw-w64/files/mingw-w64/) | 编译器。选择seh版本，并且需要将/bin添加到环境变量 |[OpenCV](https://github.com/opencv/opencv/releases)| 计算机视觉库|
|[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)|矩阵库|[PCL](https://github.com/PointCloudLibrary/pcl/releases)|处理点云库|
|[CMake](https://cmake.org/download/)|编译工具|[Git](https://git-scm.com/download/win)|版本控制系统|

# Foxmail

|         | 邮箱类型 | 收件服务器            | SSL      | 端口 | 发件服务器            | SSL      | 端口 |
| ------- | -------- | --------------------- | -------- | ---- | --------------------- | -------- | ---- |
| outlook | IMAP     | imap-mail.outlook.com | &#10003; | 993  | smtp-mail.outlook.com | STARTTLS | 587  |
| gmail   | IMAP     | imap.gmail.com        | &#10003; | 993  | smtp.gmail.com        | &#10003; | 465  |
| 126     | IMAP     | imap.126.com          | &#10003; | 993  | smtp.126.com          | &#10003; | 465  |
| qq      | IMAP     | imap.qq.com           | &#10003; | 993  | smtp.qq.com           | &#10003; | 465  |



>

