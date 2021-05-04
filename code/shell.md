## 获取指定目录下所有子目录的完整路径

```shell
#递归			筛选出目录  删除最后的冒号
ls -R ./ | grep / | sed 's/://g'
```

## 从路径中提取文件名or目录

> 参考：[Linux shell 之 提取文件名和目录名的一些方法](https://blog.csdn.net/ljianhui/article/details/43128465)

- 方法一：`basename`和`dirname`

  ```shell
  $ dir=/home/xzf/Projects/temp.txt
  # 从文件路径中提取文件名（带后缀）
  $ echo $(basename $dir)
  temp.txt
  # 从文件路径中提取文件名（不带后缀）
  $ echo $(basename $dir .txt)
  temp
  # 从（文件or文件夹的）路径中提取目录
  $ echo $(dirname $dir)
  /home/xzf/Projects
  $ echo $(dirname $(dirname $dir))
  /home/xzf
  ```

- 方法二：使用`#`和`%`

  ```shell
  $ dir=/home/xzf/Projects/temp.txt
  # 从文件路径中提取文件名（带后缀）
  $ echo ${dir##*/}
  temp.txt
  # 从（文件or文件夹的）路径中提取目录
  $ echo ${dir%/*}
  /home/xzf/Projects
  # 提取文件后缀
  $ echo ${dir##*.}
  txt
  # 更改文件后缀
  $ echo ${dir%.*}".md"
  /home/xzf/Projects/temp.md
  ```

> \#：表示从左边算起第一个
> %：表示从右边算起第一个
> \##：表示从左边算起最后一个
> %%：表示从右边算起最后一个
> *：表示要删除的内容
> ​	对于#和##的情况，它位于指定的字符（例子中的'/'和'.'）的左边，表于删除指定字符及其左边的内容；
> ​	对于%和%%的情况，它位于指定的字符（例子中的'/'和'.'）的右边，表示删除指定字符及其右边的内容。
> ​	不能把\*号放在#或##的右边，反之亦然。