- 数字前面补0

```python
# 方法一：zfill
n = "123"
s = n.zfill(5)	# "00123"
n = "-123"
s = n.zfill(5)	# "-0123"
# 方法二：格式化
n = 123
s = "%05d" % n	# "00123"
```

- [二维列表转置](https://blog.csdn.net/chichu261/article/details/102847030)

  ```python
  a = [[1, 2, 3], [4, 5, 6]]
  b = tuple(zip(*a))
  c = list(zip(*a))
  d = list(map(list, zip(*a)))
  print(b)  # ((1, 4), (2, 5), (3, 6))
  print(c)  # [(1, 4), (2, 5), (3, 6)]
  print(d)  # [[1, 4], [2, 5], [3, 6]]
  ```

  

# 库

## EasyDict

可以使得以属性的方式去访问字典的值

```python
>>> from easydict import EasyDict as edict
>>> d = edict({'foo':3, 'bar':{'x':1, 'y':2}})
>>> d.foo
3
>>> d.bar.x
1
>>> d = edict(foo=3)
>>> d.foo
3
```

解析json目录时很有用

```python
>>> from easydict import EasyDict as edict
>>> from simplejson import loads
>>> j = """{
"Buffer": 12,
"List1": [
    {"type" : "point", "coordinates" : [100.1,54.9] },
    {"type" : "point", "coordinates" : [109.4,65.1] },
    {"type" : "point", "coordinates" : [115.2,80.2] },
    {"type" : "point", "coordinates" : [150.9,97.8] }
]
}"""
>>> d = edict(loads(j))
>>> d.Buffer
12
>>> d.List1[0].coordinates[1]
```

可以利用easydict建立全局的变量

```python
from easydict import EasyDict as edict
config = edict()
config.TRAIN = edict() # 创建一个字典，key是Train,值是{}
config.Test = edict()
 # config.TRAIN = {} # 这个和上面的那句话是等价的，相当于创建一个字典的key
config.TRAIN.batch_size = 25  # 然后在里面写值,表示Train里面的value也是一个字典
config.TRAIN.early_stopping_num = 10
config.TRAIN.lr = 0.0001
```

## Json

- `json.dump`用于将 Python 对象保存到JSON文件中。语法如下

  ```python
  json.dump(obj, fp, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw)
  ```

  > `fp`：file对象，可以用`open()`创建
  >
  > `indent`：表示缩进等级，可以为非负整数或者字符串。如果为非负整数，则表示缩进的空格个数；如果为字符串 (比如 `"\t"`)，那个字符串会被用于缩进每一层。；如果为`None` (默认值) ，则为最紧凑的表达（不换行）
  >
  > `sort_keys`：为true（默认为 `False`），那么字典的输出会以键的顺序排序

  ```python
  import json
  file=open("test.json", "w", encoding='utf-8')
  test=[{"num":1, "position":[1,2,3]},
  	   {"num":2, "position":[11,22,33]}]
  json.dump(test, file, sort_keys=False, indent=4)
  ```

- `json.load`：从JSON文件读取数据并转为Python对象，语法如下

  ```python
  json.load(fp, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)
  ```

  