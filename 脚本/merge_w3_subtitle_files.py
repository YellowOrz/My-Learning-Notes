'''
@file     merge_w3_subtitle_files.py
@author   YellowOrz
@details  合并巫师3的字幕文件（.w3string），主要用于制作双语字幕
@note     首先下载编解码的工具w3strings.exe，https://www.nexusmods.com/witcher3/mods/1055
          然后准备字幕文件，格式为.w3string，比如来自路径下The Witcher 3/content/content0
@date     2025/01/01
@example  # 在游戏目录的content文件夹下（包含很多content）运行如下bash命令
          game_root="C:/Program Files (x86)/Steam/steamapps/common/The Witcher 3/content"
          w3strings_exe="C:/Users/orz/Downloads/w3strings encoder-v0.4.1.zip-1055-0-4-1/w3strings.exe"
          py_file="C:/Users/orz/Downloads/w3strings encoder-v0.4.1.zip-1055-0-4-1/merge_w3_subtitle_files.py"
          cd "$game_root"
          for i in content*; do 
            echo ====================== $i;
            cd $i;
            python.exe "$py_file" --overwrite_main --save_csv -m cn.w3strings -s en.w3strings -e "$w3strings_exe" --min_length 10;
            cd ..;
          done
'''

import os
import argparse
import logging
from itertools import zip_longest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='合并巫师3的字幕文件（.w3string），主要用于制作双语字幕')
parser.add_argument('-e','--w3strings_path', type=str, help='编解码工具w3strings.exe的路径', default='w3strings.exe')
parser.add_argument('-m','--main_subs_path', type=str, help='主字幕的路径', default='cn.w3strings')
parser.add_argument('-s','--sec_subs_path', type=str, help='次字幕的路径', default='en.w3strings')
parser.add_argument('-l', '--min_length', type=int, help='主字幕中字数小于该值的不会合并次字幕，因为UI界面中使用双语字幕会错位', default=0)
parser.add_argument('-o', '--overwrite_main', action='store_true', help='覆盖主字幕，会先备份之前的主字幕')
parser.add_argument('-c', '--save_csv', action='store_true', help='保存中途生成的csv文件')
args = parser.parse_args()

# 检查输入是否存在
if not os.path.exists(args.w3strings_path):
  logging.error("编解码工具{}不存在，请检查路径".format(args.w3strings_path))
  exit(1)
if not os.path.exists(args.main_subs_path):
  logging.error("主字幕{}不存在，请检查路径".format(args.main_subs_path))
  exit(1)
if not os.path.exists(args.sec_subs_path):
  logging.error("次字幕{}不存在，请检查路径".format(args.sec_subs_path))
  exit(1)

# 把.w3strings转.csv
main_csv = args.main_subs_path+".csv"
sec_csv = args.sec_subs_path+".csv"
if not os.path.exists(main_csv):
  # os.system(r'"{}" -d "{}"'.format(args.w3strings_path, args.main_subs_path))
  os.system(f'"{args.w3strings_path}" -d {args.main_subs_path}')
if not os.path.exists(sec_csv):
  # os.system(r'"{}" -d "{}"'.format(args.w3strings_path, args.sec_subs_path))
  os.system(f'"{args.w3strings_path}" -d {args.sec_subs_path}')
logging.info("已将.w3strings转.csv")

# 读取main
head_info = []
main_subs = {}
with open(main_csv, 'r', encoding='utf-8') as f:
  for line in f.readlines():
    line = line.strip()
    if line.startswith(";"): # 以";"开头的都是注释，统一方到一起
      head_info.append(line)
      continue
    key, value = line.rsplit("|", 1)
    main_subs[key] = value
logging.info("已读取主字幕")

# 读取sec，然后加入main
with open(sec_csv, 'r', encoding='utf-8') as f:
  for line in f.readlines():
    line = line.strip()
    if line.startswith(";"): # 以";"开头的都是注释
      continue
    key, value = line.rsplit("|", 1)
    # 合并sec的字幕和main的字幕
    if key in main_subs:
      if len(main_subs[key]) < args.min_length:
        continue
      main_strs = main_subs[key].split("<br>")
      sec_strs = value.split("<br>")
      if len(main_strs) == len(sec_strs):
        main_subs[key] = "<br>".join([main_strs[i]+"<br>"+sec_strs[i] for i in range(len(main_strs))])
      else:
        logging.warning("字幕中<br>的数量不一样：{}".format(key))
        # 不等长的list交替合并
        merge_strs = [x for pair in zip_longest(main_strs, sec_strs, fillvalue=None) for x in pair if x is not None]
        main_subs[key] = "<br>".join(merge_strs).replace("<br><br>", "<br>")  # 防止出现三个及以上连续的<br>
    else:
      logging.warning("找不到对应的key，跳过：{} {}".format(key, value))
logging.info("已合并次字幕到主字幕")

# 保存合并后的字幕到新的cvs
new_csv = main_csv + ".merged.csv"
with open(new_csv, 'w', encoding='utf-8', newline='\n') as f:
  f.write("\n".join(head_info))
  f.write("\n")
  for key, value in main_subs.items():
    f.write(key+"|"+value+"\n")
logging.info("已保存合并后的字幕到：{}".format(new_csv))

# 把.csv转.w3strings  # TODO: --id-space怎么用？
# os.system(r'"{}" -e "{}" --force-ignore-id-space-check-i-know-what-i-am-doing'.format(args.w3strings_path, new_csv))
print(f'============="{args.w3strings_path}" -e "{new_csv}" --force-ignore-id-space-check-i-know-what-i-am-doing')
os.system(f'"{args.w3strings_path}" -e {new_csv} --force-ignore-id-space-check-i-know-what-i-am-doing')
logging.info("已将.csv转.w3strings")

# 删除csv文件
if not args.save_csv:
  os.remove(main_csv)
  os.remove(sec_csv)
  os.remove(new_csv)

# 如果需要覆盖主字幕，则备份
if args.overwrite_main:
  back_main_path = args.main_subs_path+".bak"
  if os.path.exists(back_main_path):
    logging.error("主字幕的备份文件已存在，请先删除：{}".format(back_main_path))
  else:
    os.rename(args.main_subs_path, back_main_path)
    logging.info("已备份主字幕到：{}".format(back_main_path))

    os.rename(new_csv+".w3strings", args.main_subs_path)
    logging.info("已覆盖主字幕")
