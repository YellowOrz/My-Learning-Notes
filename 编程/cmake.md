# CMake Practice

> https://gavinliu6.github.io/CMake-Practice-zh-CN/#/

## 常用指令

- `PROJECT`指令：

    ```cmake
    PROJECT(projectname [CXX] [C] [Java])
    ```

    - 会隐式定义两个cmake变量`<projectname>_BINARY_DIR`和`<projectname>_SOURCE_DIR`，但名称会随着项目名变化（属于内部编译）

    > ==内部编译==：在源码所在路径执行`cmake .`
    >
    > ==外部编译==：新建文件夹build后执行`cmake ..`

- `SET`指令：显式定义变量

    ```cmake
    SET(VAR [VALUE] [CHECK TYPE DOCSTRING [FORCE]])
    ```

- `MESSAGE`指令

    ```cmake
    MESSAGE([SEND_ERROR | STATUS | FATAL_ERROR] "message to display" ...) 
    ```

    > `SEND_ERROR`，产生错误，生成过程被跳过
    >
    > `STATUS`，输出前缀为`--`的信息
    >
    > `FATAL_ERROR`，立即终止所有cmake过程

- `ADD_SUBDIRECTORY `指令：当前工程添加存放源文件的子目录，并可以指定中间二进制和目标二进制存放的位置。可以用来安装目录？

    ```cmake
    ADD_SUBDIRECTORY(source_dir [binary_dir] [EXCLUDE_FROM_ALL])
    ```

    > `source_dir`：存放源文件的子目录
    >
    > `binary_dir`：指定编译输出路径
    >
    > `EXCLUDE_FROM_ALL` ：将该目录从编译过程中排除？？？

    - 添加sub directory（例如src）之后，如果不指定编译输出路径，则默认路径为build下的同名文件夹（例如build/src）

- 【不推荐】`SUBDIRS`指令：一次添加多个子目录，并且即使外部编译（即建个build文件夹），子目录体系仍然会被保存（不能指定编译路径？？？）

    ```cmake
    SUBDIRS(dir1 dir2 ...)
    ```

- 目标文件的`INSTALL`指令：

    ```cmake
    INSTALL(TARGETS target1 target2 target3 ... [
         [ARCHIVE|LIBRARY|RUNTIME] [DESTINATION <dir>] [PERMISSIONS permissions...]
         [CONFIGURATIONS [Debug|Release|...]]
         [COMPONENT <component>] [OPTIONAL]
     ] [...])
    ```

    > target1、target2、target3就是我们ADD_EXECUTABLE 或者 ADD_LIBRARY 定义的目标文件，可能是可执行二进制、动态库、静态库。
    >
    > 目标类型：`ARCHIVE` 特指静态库，`LIBRARY` 特指动态库，`RUNTIME` 特指可执行目标二进制
    >
    > `DESTINATION` 定义安装的路径<dir>。为绝对路径（以/开头）， `CMAKE_INSTALL_PREFIX` 无效了；为相对路径， 则安装的路径就是 `${CMAKE_INSTALL_PREFIX}/<DESTINATION 定义的路径>`

- 普通文件的`INSTALL`指令：

    ```cmake
    INSTALL(FILES file1 file2 file3 ... DESTINATION <dir>
         [PERMISSIONS permissions...] [CONFIGURATIONS [Debug|Release|...]]
         [COMPONENT <component>] [RENAME <name>] [OPTIONAL]
    )
    ```

    > 不指定`PERMISSIONS`，默认为644（等价于“OWNER_WRITE, OWNER_READ, GROUP_READ, WORLD_READ”）

- 非目标文件的可执行程序（例如脚本）的`INSTALL`指令：

    ```cmake
    INSTALL(PROGRAMS files... DESTINATION <dir>
         [PERMISSIONS permissions...] [CONFIGURATIONS [Debug|Release|...]]
         [COMPONENT <component>] [RENAME <name>] [OPTIONAL]
    )
    ```

    - 跟普通文件唯一的区别是默认权限为755（等价于“OWNER_EXECUTE, GROUP_EXECUTE,  WORLD_EXECUTE”）

- 目录的`INSTALL`指令：

    ```cmake
    INSTALL(DIRECTORY dirs... DESTINATION <dir>
        [FILE_PERMISSIONS permissions...]
    	[DIRECTORY_PERMISSIONS permissions...]
     	[USE_SOURCE_PERMISSIONS]
     	[CONFIGURATIONS [Debug|Release|...]]
     	[COMPONENT <component>]
     	[[PATTERN <pattern> | REGEX <regex>]
     	[EXCLUDE] [PERMISSIONS permissions...]] [...]
    )
    ```

    > `DESTINATION`为相对路径。路径以/结尾（例如test/abc/），则这个目录将被安装 成为 目标路径下的 文件名（相当于被改名成了abc）；不以/结尾（例如test/abc），则不改名，安装到abc下面（例如test/abc/source_dir）
    >
    > `PATTERN` ：使用正则表达式进行过滤
    >
    > `PERMISSIONS` ：指定 PATTERN 过滤后的文件权限。

- 安装时 执行CMAKE 脚本的`INSTALL`指令

    ```cmake
    INSTALL([[SCRIPT <.cmake file> ] [CODE ]] [...])
    ```

    > `SCRIPT` 参数用于在安装时调用 cmake 脚本文件（也就是 `.cmake` 文件）
    >
    > `CODE` 参数用于执行 CMAKE 指令，必须以双引号括起来，例如
    >
    > ```cmake
    > INSTALL(CODE "MESSAGE(\"Sample install message.\")")
    > ```

## 基本语法

- 变量使用`${}`方式取值，但是在 IF 控制语句中是直接使用变量名。
- 指令的参数之间使用空格或分号分开。
- 指令是大小写无关的，参数和变量是大小写相关的
- ~~忽略掉 source 列表中的源文件后缀~~
- 

## 常用变量

- `<projectname>_BINARY_DIR`和 `PROJECT_BINARY_DIR`：编译文件存放路径（即新建的build文件夹路径）

    - 两者等价，前者属于内部编译（在`PROJECT`指令中隐式定义），后者属于外部编译

- `<projectname>_SOURCE_DIR`和`PROJECT_SOURCE_DIR`：源代码（或者叫工程）路径

    - 两者等价，前者属于内部编译（在`PROJECT`指令中隐式定义），后者属于外部编译

- `EXECUTABLE_OUTPUT_PATH` 和 `LIBRARY_OUTPUT_PATH` 变量：最终的目标二进制（可执行 or 库文件）的位置，例如

    ```cmake
    SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
    SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
    ```

- `CMAKE_INSTALL_PREFIX`：install路径，默认为`/usr/local`