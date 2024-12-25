"""
@file     download_upxs.py
@author   YellowOrz
@detail   批量下载utools的upxs插件
@note     pypi依赖html_to_json
@data     2024/12/15
@todo     - plugin_info包含评分、开发者的信息
          - 读取之前的plugin_info.json，只下载版本更新的插件
          - 读取之前的plugin_info.json，删除旧版本的插件
          - 添加args
@example  python download_upxs.py | tee download.log
"""
from urllib import request
from urllib.parse import quote
import string
from bs4 import BeautifulSoup
import re
import os
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: - %(message)s')

# 遍历插件分类，获取插件的部分url
# 插件列表的utl格式是https://www.u.tools/plugins/topic/xxx/，其中xxx是编号
plugin_info = {}
# plugins_part_url = []
# plugins_name = []
for id in range(1, 100):
    url = "https://www.u.tools/plugins/topic/{}/".format(id)
    try:
        res = request.urlopen(url)
    except:
        continue
    logging.info("reading {}".format(url))
    html_content = res.read().decode('utf-8')

    soup = BeautifulSoup(html_content, 'lxml')  # 或者使用 'html.parser
    topics_content = soup.find_all(class_=re.compile(r'^Topics_info'))
    for topic in topics_content:
        plugin_name = topic.find(
            class_=re.compile(r'^Topics_name')).contents[0]
        # plugins_name.append(plugin_name)
        plugin_url_part = topic.find('a').get('href')
        # plugins_part_url.append(plugin_url_part)
        plugin_info[plugin_name] = {}
        plugin_info[plugin_name]['part_url'] = plugin_url_part
        pass
    pass

# 下载插件
plugin_url_tmplate = "https://www.u.tools{}"
download_dir = "upxs"
if not os.path.exists(download_dir):
    os.mkdir(download_dir)
for plugin_name in plugin_info.keys():
    plugin_url = plugin_url_tmplate.format(
        plugin_info[plugin_name]['part_url'])
    plugin_info[plugin_name]['url'] = plugin_url
    logging.info("downloading {}".format(plugin_url))
    try:
        res = request.urlopen(quote(plugin_url, safe=':/?=&'))
    except:
        logging.error(
            "plugin {} download failed, because url is invalid".format(
                plugin_name))
        continue

    html_content = res.read().decode('utf-8')

    soup = BeautifulSoup(html_content, 'lxml')  # 或者使用 'html.parser
    topics_content = soup.find_all(class_=re.compile(r'^Details_viewerItem'))
    plugin_version = ""
    plugin_download_url = ""
    for topic in topics_content:
        if "版本" in topic.text:
            plugin_version = topic.text
            continue
        if "下载" in topic.text:
            plugin_download_url = topic.find('a').get('href')
    plugin_info[plugin_name]['version'] = plugin_version
    plugin_info[plugin_name]['download_url'] = plugin_download_url
    if plugin_version == "":
        logging.error(
            "plugin {} download failed, because version is empty".format(
                plugin_name))
        continue
    if plugin_download_url == "":
        logging.error(
            "plugin {} download failed, because download url is empty".format(
                plugin_name))
        continue
    download_name = "{}_{}.upxs".format(plugin_name, plugin_version)
    try:
        request.urlretrieve(plugin_download_url,
                            os.path.join(download_dir, download_name))
    except:
        logging.error("plugin {} download failed".format(plugin_name))
    pass

with open("plugin_info.json", "w", encoding="utf-8") as f:
    json.dump(plugin_info, f, ensure_ascii=False, indent=4)
pass
