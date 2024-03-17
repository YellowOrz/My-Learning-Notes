## Docker

- **[jeessy2/ddns-go](https://github.com/jeessy2/ddns-go)**：动态域名解析，[教程](https://www.bilibili.com/read/cv21513676/)
- [**yobasystems/alpine-mariadb**](https://hub.docker.com/r/yobasystems/alpine-mariadb)：轻量化的数据库，用来给记账软件使用，[教程](https://forum.zspace.cn/forum.php?mod=viewthread&tid=13604&highlight=firefly)
- [**fireflyiii/core**](https://hub.docker.com/r/fireflyiii/core)：记账，[教程](https://forum.zspace.cn/forum.php?mod=viewthread&tid=13604&highlight=firefly)
- **[misterbabou/gptwol](https://github.com/Misterbabou/gptwol)**：：局域网启动电脑，[github链接](https://github.com/Misterbabou/gptwol)
- [**chishin/nginx-proxy-manager-zh**](https://hub.docker.com/r/chishin/nginx-proxy-manager-zh)：反向代理，[教程](https://forum.zspace.cn/forum.php?mod=viewthread&tid=20787&highlight=nginx)

    - [设置访问密码](https://www.cnblogs.com/sueyyyy/p/10028092.html)：首先用htpasswd生成密码文件，然后在所需配置的代理中的“自定义 Nginx 配置”添加如下两行内容

        ```
        auth_basic "private server, not public"; 		# 提示信息
        auth_basic_user_file /data/passwd_file;			# 密码文件路径
        ```

- [**mzz2017/V2RayA**](https://hub.docker.com/r/mzz2017/v2raya)：代理，[自己写的教程](https://forum.zspace.cn/forum.php?mod=viewthread&tid=46701)、[官方教程](https://v2raya.org/docs/prologue/installation/docker/)（注意环境变量，创建容器d的时候启用NET_ADMIN）
- **[homeassistant/home-assistant](https://hub.docker.com/r/homeassistant/home-assistant/tags)**：智能家居，[教程1](https://www.molingran.com/p/zspace-home-assistant/)、
    - [添加米家的教程](https://github.com/al-one/hass-xiaomi-miot/blob/master/README_zh.md)
    - [添加反向代理后报错400 Bad Request](https://bbs.hassbian.com/forum.php?mod=viewthread&tid=13487)：到 配置=>系统=>日志 里查看所需IP（好像不用带子网掩码）
    - [适配国内环境的安卓客户端](https://github.com/nesror/Home-Assistant-Companion-for-Android)
- [**xiaoyaliu/alist**](https://hub.docker.com/r/xiaoyaliu/alist)：资源大全
- [shinobisystems/shinobi](shinobisystems/shinobi)：监控摄像头录像系统。[官方文档](https://gitlab.com/Shinobi-Systems/Shinobi/-/tree/dev/Docker)，第三方教程（[文档](https://post.smzdm.com/p/a259dnwp/)、[视频](https://www.bilibili.com/video/BV1f4411n73x)）

### 放弃

- **[chishin/wol-go-web](chishin/wol-go-web)**：局域网启动电脑，但是在我的NAS上运行不起来
- **[dreamacro/clash](dreamacro/clash)**：代理，[教程](https://fugary.com/?p=363)，但是不能自动更新
- **[haishanh/yacd](https://github.com/haishanh/yacd)**：clash的网页管理面板

