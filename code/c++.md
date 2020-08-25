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

- c++风格：一次读取一行

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

    

## 递归读取某一文件夹下文件/子目录的完整路径

可以指定文件类型、也可以只搜索

```cpp
#include <dirent.h>
bool GetFilesPath(string father_dir, vector<string> &files ){
    DIR *dir=opendir(father_dir.c_str());
    if(dir==NULL){
        cout << father_dir << " don't exit!" << endl;
        return false;
    }
    struct dirent *entry;
    while((entry = readdir(dir)) != NULL){
        string name(entry->d_name);               
        if(entry->d_type == DT_DIR ) { // 类型为目录
            if (name.find(".") == string::npos) {	//排除“.”和“..”以及隐藏文件夹
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

# 字符串操作

## 根据分隔符分割字符串

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
cout << RED << "I am RED. " << RESET << "I am normal. " << YELLOW << "I am YELLOW" << endl; 
```

