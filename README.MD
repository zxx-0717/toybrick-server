1、设置自启动

sudo vim /usr/local/bin/tb.local.after

添加

cd /home/toybrick/nanodet-service

sudo -u toybrick /home/toybrick/nanodet-service/boot.sh


2、更改boot.sh权限

sudo chmod 777 /home/toybrick/nanode-service/boot.sh

3、 solve sudo python3 import module error

python3

import rknnlite

print(rknnlite)


sudo python3

import sys

print(sys.path)


在sys.path中增加my.pth 文件，直接添加一句rknnlite 的path
