# CMake Practice

> https://gavinliu6.github.io/CMake-Practice-zh-CN/#/

## 常用指令

==注意：以下所有大写都是cmake中的关键词==

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
    INSTALL(TARGETS target1 target2 target3 ... 
    	[[ARCHIVE|LIBRARY|RUNTIME] [DESTINATION <dir>] [PERMISSIONS permissions...] [CONFIGURATIONS [Debug|Release|...]]
         [COMPONENT <component>] [OPTIONAL]
     ] [...])
    ```

    > target1、target2、target3就是我们ADD_EXECUTABLE 或者 ADD_LIBRARY 定义的目标文件，可能是可执行二进制、动态库、静态库。
    >
    > 目标类型：`ARCHIVE` 特指静态库，`LIBRARY` 特指动态库，`RUNTIME` 特指可执行目标二进制。可以同时填写多个
    >
    > `DESTINATION` 定义安装的路径<dir>。为绝对路径（以/开头）， `CMAKE_INSTALL_PREFIX` 无效了；为相对路径， 则安装的路径就是 `${CMAKE_INSTALL_PREFIX}/<DESTINATION 定义的路径>`

- 普通文件的`INSTALL`指令：比如头文件

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

- `ADD_LIBRARY`指令：

    ```cmake
    ADD_LIBRARY(libname [SHARED|STATIC|MODULE] [EXCLUDE_FROM_ALL]
         source1 source2 ... sourceN
    )
    ```

    > libname不用写全后缀，系统会根据库类型自动添加
    >
    > 库的类型：SHARED，动态库；STATIC，静态库；MODULE，在使用 dyld 的系统有效（如果不支持 dyld，则被当作 SHARED 对待）。默认值取决于 变量`BUILD_SHARED_LIBS`的值
    >
    > `EXCLUDE_FROM_ALL`：这个库不会被默认构建，除非有其他的组件依赖或者手工构建

    - 如果想要同时构建 相同名称的静态库和动态库，不能用两个ADD_LIBRARY，因为 target 名称是唯一的，要用`SET_TARGET_PROPERTIES`指令

- `SET_TARGET_PROPERTIES`指令：用来设置输出的名称，对于动态库，还可以用来指定动态库版本和 API 版本。

    ```cmake
    SET_TARGET_PROPERTIES(target1 target2 ...
     PROPERTIES prop1 value1
     prop2 value2 ...
    )
    ```

    - 同时构建 相同名称的静态库和动态库，例如

        ```cmake
        SET_TARGET_PROPERTIES(hello_static PROPERTIES OUTPUT_NAME "hello")
        ```

- `GET_TARGET_PROPERTY`指令：TARGET_PROPERTY是什么？？？

    ```cmake
    GET_TARGET_PROPERTY(VAR target property)
    # 例如
    GET_TARGET_PROPERTY(OUTPUT_VALUE hello_static OUTPUT_NAME)
    ```

    > 如果property没有定义，则返回 NOTFOUND

- `INCLUDE_DIRECTORIES`指令：向工程添加多个特定的头文件搜索路径

    ```cmake
    INCLUDE_DIRECTORIES([AFTER|BEFORE] [SYSTEM] dir1 dir2 ...)
    ```

    >  `AFTER` 或者 `BEFORE` 参数：控制放到当前的头文件搜索路径 前面or后面。不加这俩参数的话，默认添加到后面。也可以将变量`CMAKE_INCLUDE_DIRECTORIES_BEFORE`设置为on，将添加的头文件搜索路径放在已有路径的前面
    >
    > `SYSTEM`参数：把指定目录当成系统的搜索目录

- `LINK_DIRECTORIES`指令：添加非标准的共享库搜索路径，比如，在工程内部同时存在共享库和可执行二进制，在编译时就需要指定一下这些共享库的路径

    ```cmake
    LINK_DIRECTORIES(directory1 directory2 ...)
    ```

- `TARGET_LINK_LIBRARIES`指令

    ```cmake
    TARGET_LINK_LIBRARIES(target library1 <debug | optimized> library2 ...)
    ```

    > library1可以加.so或者.a

- `ADD_DEFINITIONS`指令：向C，C++编译器添加`-D`定义，例如

    ```cmake
    ADD_DEFINITIONS(-DENABLE_DEBUG -DABC)	# 参数之间用空格分割
    # 代码中定义了#ifdef ENABLE_DEBUG #endif，这个代码块就会生效
    ```

    - 若要添加其他的编译器开关，可以通过` CMAKE_C_FLAGS `变量和 `CMAKE_CXX_FLAGS` 变量设置

- `ADD_DEPENDENCIES`指令：定义 target 依赖的其他 target

    ```cmake
    ADD_DEPENDENCIES(target-name depend-target1 depend-target2 ...)
    ```

    - 确保在编译本 target 之前，其他的 target 已经被构建

- `ENABLE_TESTING` 指令：控制 Makefile 是否构建 test 目标，一般放在主CMakeLists.txt 中

    ```cmake
    ENABLE_TESTING()
    ```

- `ADD_TEST` 指令

    ```cmake
    ADD_TEST(testname Exename arg1 arg2 ...)
    ```

    > testname ：自定义的 test 名称
    >
    > Exename ：可以是构建的目标文件也可以是外部脚本等等
    >
    > arg：传递给可执行文件的参数

    - 没有在同一个 CMakeLists.txt 中打开ENABLE_TESTING()指令（<u>顺序无所谓</u>），任何 ADD_TEST 都是无效的
    - 生成Makefile后，运行make test来执行测试

- `AUX_SOURCE_DIRECTORY`指令：发现一个目录下所有的源代码文件并将列表存储在一个变量中，这个指令临时被用来自动构建源文件列表

    ```cmake
    AUX_SOURCE_DIRECTORY(dir VARIABLE)
    ```

- `CMAKE_MINIMUM_REQUIRED`指令：

    ```cmake
    CMAKE_MINIMUM_REQUIRED(VERSION versionNumber [FATAL_ERROR])
    ```

- `EXEC_PROGRAM`指令：在 CMakeLists.txt 处理过程中执行命令。不会在生成的 Makefile 中执行

    ```cmake
    EXEC_PROGRAM(Executable [directory in which to run]
                     [ARGS <arguments to executable>]
                     [OUTPUT_VARIABLE <var>]
                     [RETURN_VALUE <var>])
    # 例如，在 src 目录执行 ls 命令，并把结果和返回值存下来
    EXEC_PROGRAM(ls ARGS "*.c" OUTPUT_VARIABLE LS_OUTPUT RETURN_VALUE LS_RVALUE)
    ```

    > OUTPUT_VARIABLE 和 RETURN_VALUE 获取输出和返回值

- `FILE` 指令：文件操作

    ```cmake
    FILE(WRITE filename "message to write"... )
            FILE(APPEND filename "message to write"... )
            FILE(READ filename variable)
            FILE(GLOB  variable [RELATIVE path] [globbing expressions]...)
            FILE(GLOB_RECURSE variable [RELATIVE path]
                 [globbing expressions]...)
            FILE(REMOVE [directory]...)
            FILE(REMOVE_RECURSE [directory]...)
            FILE(MAKE_DIRECTORY [directory]...)
            FILE(RELATIVE_PATH variable directory file)
            FILE(TO_CMAKE_PATH path result)
            FILE(TO_NATIVE_PATH path result)
    ```

- `INCLUDE` 指令：载入 CMakeLists.txt 文件 或者 预定义的 cmake 模块

    ```cmake
    INCLUDE(file1 [OPTIONAL]) 
    INCLUDE(module [OPTIONAL]) 
    ```

    > OPTIONAL：文件不存在也不会产生错误。

- FIND指令

    ```
    FIND_FILE(<VAR> name1 path1 path2 ...) 		# VAR 变量代表找到的文件全路径，包含文件名
    FIND_LIBRARY(<VAR> name1 path1 path2 ...) 	# VAR 变量表示找到的库全路径，包含库文件名
    FIND_PATH(<VAR> name1 path1 path2 ...) 		# VAR 变量代表包含这个文件的路径
    FIND_PROGRAM(<VAR> name1 path1 path2 ...) 	# VAR 变量代表包含这个程序的全路径
    FIND_PACKAGE(<name> [major.minor] [QUIET] [NO_MODULE]
                     [[REQUIRED|COMPONENTS] [componets...]])
    ```

    > FIND_PACKAGE用来调用预定义在 `CMAKE_MODULE_PATH 下的 Find<name>.cmake` 模块，你也可以自己 定义`Find<name>`模块，通过`SET(CMAKE_MODULE_PATH dir)`将其放入工程的某个目录 中供工程使用

## 控制指令

- `IF`指令

    ```cmake
    IF(expression)
              # THEN section.
              COMMAND1(ARGS ...)
              COMMAND2(ARGS ...)
              ...
    ELSE(expression)
              # ELSE section.
              COMMAND1(ARGS ...)
              COMMAND2(ARGS ...)
              ...
    ENDIF(expression)
    ```

    - 表达式的使用方法
        - 


## 基本语法

- 变量使用`${}`方式取值，但是在 IF 控制语句中是直接使用变量名。
- 指令的参数之间使用空格或分号分开。
- 指令是大小写无关的，参数和变量是大小写相关的
- ~~忽略掉 source 列表中的源文件后缀~~
- 

## 常用变量

- `<projectname>_BINARY_DIR`和 `PROJECT_BINARY_DIR`和`CMAKE_BINARY_DIR`：编译文件存放路径（即新建的build文件夹路径）

    - 两者等价，前者属于内部编译（在`PROJECT`指令中隐式定义），后者属于外部编译

- `<projectname>_SOURCE_DIR`和`PROJECT_SOURCE_DIR`和`CMAKE_SOURCE_DIR`：源代码（或者叫工程）路径

    - 两者等价，前者属于内部编译（在`PROJECT`指令中隐式定义），后者属于外部编译

- `EXECUTABLE_OUTPUT_PATH` 和 `LIBRARY_OUTPUT_PATH` 变量：最终的目标二进制（可执行 or 库文件）的位置，例如

    ```cmake
    SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
    SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
    ```

- `CMAKE_INSTALL_PREFIX`：install路径，默认为`/usr/local`

- `CLEAN_DIRECT_OUTPUT`：？？？

- `CMAKE_INCLUDE_DIRECTORIES_BEFORE`：设置为on，将添加的头文件搜索路径放在已有路径的前面

- `CMAKE_CURRENT_SOURCE_DIR`：当前处理的 CMakeLists.txt 所在的路径

- `CMAKE_CURRENT_BINARY_DIR`：编译路径下当前CMakeLists.txt 对应的子路径

    - `ADD_SUBDIRECTORY`指令中的binary_dir可以更改这个变量的值
    - 使用 `SET(EXECUTABLE_OUTPUT_PATH <新路径>)` 不会对这个变量造成影响，它仅修改了最终目标文件存放的路径

- `CMAKE_CURRENT_LIST_FILE`和`CMAKE_CURRENT_LIST_LINE`：输出调用这个变量的 CMakeLists.txt 的完整路径 和 所在行

- `CMAKE_MODULE_PATH`：定义自己的 cmake 模块所在的路径

- `EXECUTABLE_OUTPUT_PATH` 和 `LIBRARY_OUTPUT_PATH`：分别用来重新定义最终结果的存放目录

- 主要开关选项：

    - `MAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS`：开启后，`ELSE`和`ENDIF`的括号中不用写东西
    - `BUILD_SHREAD_LIBS`：控制默认的库编译方式，不设置的话 `ADD_LIBRARY`默认为静态的
    - `CMAKE_C_FLAGS`：设置C编译选项，也可以通过指令`ADD_DEFINITIONS()`添加。
    - `CMAKE_CXX_FLAGS`：设置C++编译选项，也可以通过指令`ADD_DEFINITIONS()`添加。

## 常用环境变换

- `CMAKE_INCLUDE_PATH`和`CMAKE_LIBRARY_PATH`：用来弥补系统环境变量中没有包含的路径，在`FIND_PATH`和`FIND_LIBRARY`指令中会用到。需要在bash中使用export设置

- `CMAKE_INCLUDE_CURRENT_DIR`：自动添加` CMAKE_CURRENT_BINARY_DIR` 和 `CMAKE_CURRENT_SOURCE_DIR `到当前处理的 CMakeLists.txt。相当于在每个 CMakeLists.txt 加入：

    ```cmake
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    ```

- `CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE`：将工程提供的头文件目录始终至于系统头文件目录的前面，当你定义的头文件确实跟系统发生冲突时可以提供一些帮助。

## 技巧

- 同时构建 相同名称的静态库和动态库

    ```cmake
    ADD_LIBRARY(hello STATIC ${LIBHELLO_SRC})
    SET_TARGET_PROPERTIES(hello_static PROPERTIES OUTPUT_NAME "hello")
    SET_TARGET_PROPERTIES(hello PROPERTIES CLEAN_DIRECT_OUTPUT 1)
    SET_TARGET_PROPERTIES(hello_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
    ```

- 实现动态库版本号

    ```cmake
    SET_TARGET_PROPERTIES(libname PROPERTIES VERSION 1.2 SOVERSION 1)
    # 完成后得到类似下面的文件
    # libhello.so.1.2
    # libhello.so ->libhello.so.1
    # libhello.so.1->libhello.so.1.2
    ```

    > VERSION 指代动态库版本，SOVERSION 指代 API 版本

- 看 `make` 构建的详细过程，可以使用 `make VERBOSE=1` 或者 `VERBOSE=1 make` 命令来进行构建。
- cmake 使用环境变量
    - 调用系统的环境变量：`$ENV{NAME}` 
    - 设置环境变量：`SET(ENV{变量名} 值)`