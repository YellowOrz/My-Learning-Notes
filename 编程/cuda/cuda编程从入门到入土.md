

# cuda编程从入门到入土

## p3 - 显存分配（1）



## p7 - cuda性能分析

-  性能分析：运行如下命令。Win系统，同路径下找到.qdrep文件双击打开，即可查看

    ```shell
    nsys profile --stats=true <exe>
    ```

## p8 - 获取GPU属性

```c
int id;
cudaGetDevice(&id);

cudaDeviceProp props;
cudaGetDeviceProperties(&props, id);

printf("device id: %d\nsms: %d\ncapability: %d.%d\nwarp size: %d\n", id, props.multiProcessorCount, props.major, props.minor, props.warpSize);
```

## p9 - 显存分配（2）

- 使用`cudaMallocManaged()`还没有具体分配到cpu或者GPU，只有第一次使用的时候，遇到页错误，才会具体分配

