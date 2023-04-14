# Docker

> 参考资料：[Docker — 从入门到实践](http://docker-practice.github.io/zh-cn)

## 使用Docker镜像

- 获取镜像：从镜像仓库（[Docker Hub](https://hub.docker.com/search?q=&type=image)）获取镜像

    ```bash
    docker pull [参数] [DockerRegistry地址[:端口号]/]仓库名[:标签]
    # 例如
    docker pull pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
    ```

- 镜像的唯一标识是其 ID 和摘要，可以有多个标签

- 列出镜像：包含了 `仓库名`、`标签`、`镜像 ID`、`创建时间` 以及 `所占用的空间`。

    ```bash
    docker image ls
    ```

    - 所占用的空间c是解压后的大小，大于镜像仓库，而且不是实际硬盘消耗，∵docker 镜像是多层存储结构，并且可以继承、复用，同样的层只会存一遍（∵用了UnionFS）

    - 镜像 ID 则是镜像的唯一标识，一个镜像可以对应多个标签
    - 虚悬镜像(dangling image) ：仓库名和标签均为 `<none>`，∵镜像更新后，新旧镜像同名，会将旧的取消
    - 只显示顶层镜像，加参数-a可以显示中间层镜像（为了加速镜像构建、重复利用资源），同样没有标签

- 查看镜像、容器、数据卷所占用的空间

    ```bash
     docker system df
    ```

- 删除镜像：本质上是在删除某个ID的镜像

    ```bash
    docker image rm [选项] 镜像短ID|镜像长ID|镜像名|镜像摘要
    # 删除所有仓库名为 redis 的镜像
    docker image rm $(docker image ls -q redis)
    # docker image rm $(docker image ls -q redis)
    docker image rm $(docker image ls -q -f before=mongo:3.2)
    ```

## 操作Docker容器

- 新建并启动容器：退出必须用快捷键ctrl+d，使用exit命令会导致容器停止

    ```bash
    docker run [参数] 仓库:标签|镜像ID [命令]
    # 例如
    docker run -it pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime bash
    docker run -it 385dc33ab519 bash
    ```

    > 参数：
    >
    > - `-i`：交互式操作
    > - `-t` ：终端
    > - `--rm`：容器退出后随之将其删除。可以避免浪费空间
    > - `-p [p1:p2]`：将本地的p1端口映射到docker的p2端口
    > - `-P`：随机
    > - `-d`：不会把输出的结果 (STDOUT) 打印到宿主机上面(输出结果可以用 `docker container logs` 查看)

- 查看容器：显示运行中容器的状态

    ```bash
    docker container ls 
    ```

    - 加参数-a会显示终止了的容器

- 启动已终止容器

    ```bash
    docker container start 容器ID|容器名称
    ```

- 终止容器

    ```bash
    docker container stop 容器ID|容器名称
    ```

- 删除容器

    ```bash
    docker container rm 容器ID|容器名称
    ```

- 重新进入容器

    - 方法一：根run一样，不能用exit退出

        ```bash
        docker attach 容器ID|容器名称
        ```

    - （推荐）方法二：可以用exit退出

        ```bash
        docker extc [参数] 容器ID|容器名称
        ```

- 导出容器快照：可以保存在容器里面做的操作（比如安装的软件，添加的文件等）

    ```bash
    docker export 容器ID|容器名称 > 文件名
    # 例如
    docker export 7691a814370e > ubuntu.tar
    ```

- 导入容器快照

    ```
    cat 文件名 | docker import - 仓库:标签
    # 例如
    cat ubuntu.tar | docker import - test/ubuntu:v1.0
    ```

    也可以从URL或者制定目录导入

    ```
    $ docker import URL链接|目录 仓库:标签
    ```

    