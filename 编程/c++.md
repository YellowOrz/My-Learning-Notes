# 基础

## static

- [类中的**静态成员变量**](http://c.biancheng.net/view/2227.html)：
    - 实现多个对象 <u>共享数据</u>的目标。
    - 静态成员变量 是<u>全局变量</u>，不占用对象的内存，而是在所有<u>对象之外开辟内存</u>，即使<u>不创建对象也可以访问</u>，到程序结束时才释放
    - 静态成员变量必须初始化，<u>不能在类定义里边初始化</u>，只能在class body外初始化
    - 静态成员变量可以<u>通过 对象名 or 类名 访问</u>，但要遵循 private、protected 和 public 关键字的访问权限限制
- [类中的**静态成员函数**](http://c.biancheng.net/view/2228.html)：
    - 静态成员函数 <u>没有this指针</u>，编译器不会为它增加形参 this，不能调用普通成员函数，<u>只能访问静态成员（主要目的）</u>
- 

## 指针

- `void * `表示void类型的指针，可以指向任意类型的数据，或者叫做**无类型指针**，它只记录地址。
    - 支持的操作：
        - 与另一个指针进行比较；
        - 向函数传递void\*指针或从函数返回void\*指针；
        - 给另一个void*指针赋值
        - 任何类型的指针都可以直接赋值给void*指针
    - 不支持：
        - 操作void *指针所指向的对象（经过强制类型转换就可以了）
        - 直接给其他非void *指针赋值（经过强制类型转换就可以了）
    - 示例： malloc 函数返回的指针就是 void \* 型
- 

# 多线程

## thread

- [函数**形参为引用**时](https://stackoverflow.com/questions/65358028/c-thread-error-static-assert-failed-due-to-requirement)，需要将实参使用`std::ref()`转换成`reference_wrapper`后再传入，例如

    ```c++
    void foo(int &args){
    	// something
    }
    int main(){
        int args;
        std::thread worker(foo, std::ref(args));
        worker.join();
    }
    ```

- [调用**重载的类成员函数**](https://blog.csdn.net/OTZ_2333/article/details/125736959)

    ```c++
    #include<iostream>
    #include<thread>
    using namespace std;
    
    class Print{
     public:
      void print() { cout << "void print()" << 0 << endl; }
      void print(int i) { cout << "void print(int i)" << i << endl; }
      int print(float i) { cout << "void print(float i)" << i << endl; return 0; }
      void print(int i, float j) { cout << "void print(int i, float j)" << i << " " << j << endl; }
      void print(const int& i) { cout << "void print(const int& i)" << i << endl; }
    };
    int main() {
      Print p1, p2;
      thread t1(static_cast<void(Print::*)()>(&Print::print), &p1);
      thread t2(static_cast<void(Print::*)(int)>(&Print::print), &p1, 1);
      thread t3(static_cast<int(Print::*)(float)>(&Print::print), &p1, 2.0);
      thread t4(static_cast<void(Print::*)(int, float)>(&Print::print), &p1, 3, 4.0);
      thread t5(static_cast<void(Print::*)(const int&)>(&Print::print), &p2, 5); // 不能去掉&，否则调用的是void print(int i)
    
      t1.join();t2.join();t3.join();t4.join();t5.join();
      return 0;
    }
    ```

- [函数的形参即使有**默认值**，在thread中调用时仍然需要给那个参数赋值](https://stackoverflow.com/a/65182257/11271721)，因为正常调用函数时，编译器其实是将默认参数填进去了；而thread是一个标准库中的函数，不会多加一个参数，必须要求找到参数数量相同的函数

# 文件操作

## 文件读写

- c语言风格

    ```c
    FILE *fp;
    string path = "./depth.txt";
    fp = fopen(path.c_str(), "r"); //写入文件的话就是"w"，可以根据是否为二进制文件选择添加“b”
    int id=0;
    uint16_t depth[WIDTH * HEIGHT] = {0};
    fread(&id, sizeof(int), 1, fp);	//第二个参数为一次读取数据大小，必须与写入的大小保持一致；第三个参数为读取次数
    fread(depth, sizeof(uint16_t), WIDTH * HEIGHT, fp);	//读取顺序必须与写入顺序一致（因此在写入文件的时候，某种程度上就是对文件加密了，因为只要读取顺序、方式不对，读取出来的内容就不是原来的内容了）。fread执行完以后会自动将指针指向未读取的第一个元素（但是win上有时候会有bug，不会自动挪动指针）
    fclose(fp);
    ```

- c++风格：一次读取一行。头文件`#include <fstream>`

    ```c++
    string filepath = "./depth.txt", data;
    ifstream fp(filepath);
    if (!fp) {
        cout << "Error: " << "file " << filepath <<" does not exit!" << endl;
        exit(0);
    }
    while (getline(fp, data)) {
    	...
    }
    fp.close();
    ```

## 读取整个文件到string

```c++
#include <string>
#include <fstream>

std::ifstream file("file.txt");
std::string str((std::istreambuf_iterator<char>(file)),
                std::istreambuf_iterator<char>());

```

## 递归读取某一文件夹下文件/子目录的完整路径

可以指定文件类型、也可以只搜索

```cpp
#include <dirent.h>
# 注意father_dir要以“/”结尾，例如“../binfiles/”
bool GetFilesPath(const string& father_dir, vector<string> &files ){
    DIR *dir=opendir(father_dir.c_str());
    if(dir==nullptr){
        cout << father_dir << " don't exit!" << endl;
        return false;
    }
    struct dirent *entry;
    while((entry = readdir(dir)) != nullptr){
        string name(entry->d_name);               
        if(entry->d_type == DT_DIR ) { // 类型为目录
            if (name.find('.') == string::npos) {	//排除“.”和“..”以及隐藏文件夹
                string son_dir = father_dir + name + "/";

                vector<string> tempPath;
                GetFilesPath(son_dir, tempPath);
                files.insert(files.end(), tempPath.begin(), tempPath.end());
            }
        }
        else if(name.find("bin") != string::npos) // 类型为文件。只找bin文件
            files.push_back(father_dir+name);
    }
    closedir(dir);
    return true;
}
```

## 获得文件格式

```c++
std::string fn = "filename.conf";
if(fn.substr(fn.find_last_of(".") + 1) == "conf") {
	std::cout << "Yes..." << std::endl;
} else {
	std::cout << "No..." << std::endl;
}
```



# 字符串操作

## 提取字符串最后一个字符

```c++
string str = "...";
// 方法一
cout << str[str.length()-1] << endl;

// 方法二
string::const_iterator it = str.end();
it--;
cout << *it << endl;
```



## String <--> int、float、double

```c++
// int、float、double  --> string
string s = to_string(num);

// string --> int、float、double
cout << stoi(s) << endl; //string --> int;
cout << stol(s) << endl; //string --> int
cout << stof(s) << endl; //string --> float
cout << stof(s) << endl; //string --> doubel
```



## find()：查询是否包含字符(串)

```c++
string s = "...";
if (s.find("...") != String::npos)
```



## strtok()：根据分隔符分割字符串

> 参考资料：[C++之split字符串分割](https://blog.csdn.net/Mary19920410/article/details/77372828)

```cpp
#incude <vector>
vector<string> split(const string &str, const string &delim) {
    vector<string> res;
    if ("" == str) return res;
    //先将要切割的字符串从string类型转换为char*类型
    char *strs = new char[str.length() + 1]; //不要忘了
    strcpy(strs, str.c_str());

    char *d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d); // 第一个参数为起始位置
    while (p) {
        string s = p; //分割得到的字符串转换为string类型
        res.push_back(s); //存入结果数组
        p = strtok(NULL, d);
    }
    return res;
}
```

## cout彩色输出

> 参考资料：[std::cout彩色输出](https://blog.csdn.net/zww0815/article/details/51275262)

```c++
#define RESET   "\033[0m"
#define RED     "\033[31m" 
#define YELLOW  "\033[33m" 
#define BLUE    "\033[34m"
#define GREEN   "\033[32m" 
cout << RED << "I am RED. " << RESET << "I am normal. " << YELLOW << "I am YELLOW" << endl; 
```

## 多次重复一个字符

在python中可以使用`"="*10`得到`"=========="` 的效果，c++中则使用`std::string(10, '=')`

# STL

## vector

- vector.push_back中完成的是值拷贝，而不仅仅是地址的复制。

- [vector中使用emplace_back代替push_back](http://c.biancheng.net/view/6826.html)，因为emplace_back的效率更高
    - push_back() 向容器尾部添加元素时，首先会创建这个元素，然后再将这个元素拷贝或者移动到容器中（如果是拷贝的话，事后会自行销毁先前创建的这个元素）
    - emplace_back() 在实现时，则是直接在容器尾部创建这个元素，省去了拷贝或移动元素的过程。

- 删除vector中满足条件的元素

    ```c++
    std::vector<T> v;
    auto iter = v.begin();
    while (iter != v.end()) {
        if(/*条件*/)
            iter = v.erase(iter);
        else
            iter++;
    }
    ```

    

## pair

- [`std::pair`](https://www.cnblogs.com/nimeux/archive/2010/10/05/1844191.html)主要的作用是将两个数据组合成一个数据，两个数据可以是同一类型或者不同类型

    - 例如std::pair<int,float> 。

    - pair`实质上是一个结构体，其主要的两个成员变量是first和second，这两个变量可以直接使用。初

    - 始化一个pair可以使用构造函数，也可以使用std::make_pair函数，

    - make_pair函数的定义如下：

        ```c++
        template pair make_pair(T1 a, T2 b) { return pair(a, b); }
        ```

    - 一般make_pair都使用在需要pair做参数的位置，可以直接调用make_pair生成pair对象。 

    - pair可以接受隐式的类型转换，这样可以获得更高的灵活度

# 其他

## 使用PCL解析命令行参数

```c++
int main(int argc, char **argv) {
    cout << RED << "解析命令行参数中。。。" << RESET << endl;
    string parameter;
    int arg_index = pcl::console::find_argument(argc, argv, "--filename");
    if (arg_index != -1)
        parameter = argv[arg_index];
    else parameter = "...";
    ...
}
```

## 使用宏定义来定义“函数”

如果需要多次用到某个小操作，可以使用宏定义来定义一个临时的"函数"，而不是定义专门的函数（有调用时间）or 内联函数（太复杂了）

例如在ORBSLAM2中使用宏定义给二维向量进行旋转

![image-20220601164910355](images/image-20220601164910355.png)

## 获取屏幕分辨率

- Linux：头文件不需要在CMake中进行任何设置

    ```c++
    #include <X11/Xlib.h>	// 直接include
    
    Display* d = XOpenDisplay(NULL);
    Screen* s = DefaultScreenOfDisplay(d);
    cout << "Witdh = " << s->width << ", height = " << s->height << endl;
    ```


# CMake

- 设置c++版本

    ```cmake
    # 方法一
    add_compile_options(-std=c++11)
    # 方法二: 从CMake v3.1开始
    set(CMAKE_CXX_STANDARD 14)
    # 方法三
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    ```

    

# CPU眼里的C++

> [CPU眼里的：编程知识](https://space.bilibili.com/261582436/channel/collectiondetail?sid=59030)

## main函数

- main函数实现的汇编跟普通函数一样

- main是默认的入口，但是在gcc中添加参数`-e`指定入口，很多嵌入式平台可以在代码中指定C程序的入口

## 变量

- 每个变量对应一个内存地址，其类型决定了占用长度

## i++ vs i++

- 简单调用没有任何的区别
- 涉及到赋值，后加需多一个拷贝操作，对于对象、iter用前加
- 前加和后加都是函数调用，都有return
- 后加return的是变量原始值，可以当做是独立的常量
- 前加return的是变量的引用（指针）

![image-20230418125140656](images/image-20230418125140656.png)

![image-20230418125155967](images/image-20230418125155967.png)



## this指针

- 成员函数和普通函数完全等价，this指针就是一个被隐藏的普通参数

- 对象调用成员函数就是把自己的地址作为this指针传入，只是被c++语法隐藏了

## 构造函数

- 构造函数跟普通成员函数的汇编一样，也有隐藏参数this指针

- 派生类的构造函数会多调用基类的构造函数

- 如果存在虚函数，构造函数中还会记录虚函数表地址（无论是否有继承，都只记录自己的虚函数，具体见下面的“虚函数”），从而记录在对象中

## 虚函数

- 虚函数 实现的汇编跟 普通函数 一样

- 调用的时候，普通函数的地址已经确定（静态绑定），而虚函数的地址要根据寄存器确定（动态绑定）

- 当 类 包含虚函数，会偷偷生成一个隐藏成员变量（称为v指针，不是this指针），在构造函数中初始化为**虚函数表地址**；再根据 表中偏移 就可以得到**虚函数地址**

    - ∴ 构造函数中不能调用虚函数，以免搞混

- 虚函数的出现大大降低了函数指针的使用率

## 多态

- 只要做 类型转换 都是不安全，编译器都会warn，除了多态

- 多态中 常用 基类指针 指向 派生类（即派生类 降级成 基类，相当于 **代码复用**），但是不能反过来（∵有非法内存）

    ```cpp
    class A{...}
    class B: public A{...}
    A a;
    B b;
    A *pa = &b;  // good
    B *pb = &a;  // bad
    ```

- 多态 通过 虚函数 扩展 派生类的特性

## 指针变量 | 数组指针 | 野指针

- 无论什么类型，**指针变量的读写** 跟 普通变量 **一样**

- \*操作就是对内存操作

- 对指针做+、-操作就是在做内存偏移

## 指针 | 万物皆“指针”

- 普通变量 也可以做指针操作，∵ 变量 是 内存地址 的别名

    ```cpp
    int a = 0;  // a address = 0x1234
    // 如下几种方式等价
    *(int*)&a=1;
    *(int*)&0x1234a=1;
    a = 0;
    ```

    ```cpp
    class A{int x;}
    A a;
    // 如下几种方式等价
    a.x = 1;
    (&a)->x = 1;
    ```

### 参数传递 | 传值 vs 传指针 vs 传引用

- 传递参数，就是在给寄存器赋值

# C++ 11

## 原始字面量

- 定义：`R"xxx(原始字符串)xxx"`，其中xxx表示备注（要求括号前后的备注一样，不会被输出，可以省略），如果xxx省略了则括号也可以省略
- 用了原始字面量，可以去除多行字符串中的反斜杠

## 指针空值类型 - nullptr

- 在c++中，NULL为0（在c中为`(void *)0`），∵`void *`不能隐式转换成其他类型的指针
- nullptr 专用于初始化空类型指针，不同类型的指针变量都可以使用 nullptr 来初始化

# 多线程和线程同步

> https://subingwen.cn/linux/thread/

## 线程概述

- Linux上，线程是轻量级的进程，所以消耗资源比进程少
- 进程有自己独立的地址空间，多个线程共用同一个地址空间
- 线程是程序的最小执行单位，进程是操作系统中最小的资源分配单位
- CPU 的调度和切换：线程的上下文切换比进程要快的多
- 线程更加廉价，启动速度更快，退出也快，对系统资源的冲击小
- 优先使用多线程，为了效率高
    - 文件 IO 操作：线程的个数 = 2 * CPU 核心数
    - 处理复杂算法：线程的个数 = CPU 的核心数

## 创建（子）线程

# 设计原则



# 设计模式

> [合集·和子烁一起五分钟学设计模式](https://space.bilibili.com/59546029/channel/collectiondetail?sid=266320)

## 单例模式

> [五分钟学设计模式.01.单例模式](https://www.bilibili.com/video/BV1af4y1y7sS)

- 定义：确保一个类只有一个实例，而且自行实例化并向整个系统提供这个实例。

- 特点：

    - 构造函数是私有的
    - 唯一的实例是当前类的静态成员变量，static修饰
    - 通过一个静态成员函数 向外界提供实例。在懒汉式中，函数内部最好加锁，防止多线程中被多次实例化
    - 函数内部最好加锁（更好的是使用双重检查锁，即加锁、然后判断是否有为空），防止多线程中被多次实例化 
    - 在实例申明中添加关键词voliate，防止"A线程分配内存、但是还没初始化的情况下，B线程直接返回"的情况

    - 一般无状态，以工具类的形式提供，防止多线程中数据冲突的问题

- 应用：序列号生成器、Web页面的计数器，或者其他创建对象消耗很多资源（IO、数据库等）的情况

- 分类：

    - 饿汉式单例模式：在类加载的时候（不是在静态成员函数中）就进行实例化
    - 懒汉式单例模式：在第一次使用的时候（在静态成员函数中）进行实例化。

## 简单工厂模式(Simple Factory Pattern)

> [五分钟学设计模式.02.简单工厂模式](https://www.bilibili.com/video/BV1Ta4y1Y7af)

- 定义：可以根据参数的不同返回不同类的实例。即专门定义一个类来负责创建其他类的实例，被创建的实例通常都具有共同的父类。
    - 又称为静态工厂方法(Static Factory Method)模式，它属于类创建型模式。
- 优点：实现对象的创建和使用分离。客户端不用管则呢么创建，只 关心怎么使用
- 缺点：不灵活，没有满足开闭原则，要新增产品就要修改工厂类（判断逻辑之类的）

## 工厂模式

- 定义：定义一个用于创建对象的接口，让子类决定实例化哪个类。工厂模式使一个类的实例化延迟到其子类。
