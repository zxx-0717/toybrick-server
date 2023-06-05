# -*- encoding==utf-8 -*-
import socket
import time
from rknnlite.api import RKNNLite
import cv2
import math
import numpy as np
from utils import *

# ---------------------------------初始化rknn---------------------------------

RKNN_PATH = r'/home/toybrick/nanodet-service/nanodet-plus-m_416_torchscript.rknn'
#RKNN_PATH = r'/home/toybrick/nanodet-service/nanodet_m_416_torchscript.rknn'

# 创建 RKNN 对象
print('------正在创建RKNN------')
rknn = RKNNLite(True)
# 从当前目录加载 RKNN 模型 resnet_18
print('------正在加载rknn模型------')
ret = rknn.load_rknn(path=RKNN_PATH)
if ret != 0:
        print('------模型加载失败!------')
        exit(ret)
# 初始化运行时环境，设置目标开发板为 RK1808
# 如果只有一个设备，device_id 可以不填
print('------正在加载init_runtime------')
ret = rknn.init_runtime()#target='rk1808', device_id='1808'
if ret != 0:
        print('------init_runtime失败!------')
        exit(ret)
# ---------------------------------初始化rknn---------------------------------


# ---------------------------------初始化TCP---------------------------------
HOST = ''
PORT = 8080
ADDRESS = (HOST, PORT)
# 创建一个套接字
print('------正在创建TCP------')
tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 绑定本地ip
tcpServer.bind(ADDRESS)
# 开始监听
tcpServer.listen(5)
print('------TCP创建完毕------')
# ---------------------------------初始化TCP---------------------------------

while True:
    print('------等待TCP客户端连接------')
    client_socket, client_address = tcpServer.accept()
    print('------TCP客户端连接成功------')
    try:
        while True:
            # 接收标志数据
            data = client_socket.recv(1024)
            if data:
                # 通知客户端“已收到标志数据，可以发送图像数据”
                client_socket.send(b"ok")
                # 处理标志数据
                flag = data.decode().split(",")
                # 图像字节流数据的总长度
                total = int(flag[0])
                # 接收到的数据计数
                cnt = 0
                # 存放接收到的数据
                img_bytes = b""

                while cnt < total:
                    # 当接收到的数据少于数据总长度时，则循环接收图像数据，直到接收完毕
                    data = client_socket.recv(256000)
                    img_bytes += data
                    cnt += len(data)
                    print("receive:" + str(cnt) + "/" + flag[0])
                #

                # 解析接收到的字节流数据，并显示图像
                img = np.asarray(bytearray(img_bytes), dtype="uint8")
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)

                # 预处理图片
                # srcimg = cv2.imread(r'/home/toybrick/nanodet-service/img3.jpg')
                img, newh, neww, top, left = pre_process(img)
                # 调用 inference 接口进行推理
                start = time.perf_counter()
                outputs = rknn.inference(inputs=[img],data_type='float32',data_format="nchw")[0]
                start1 = time.perf_counter()
                print("延时：" + str(int((start1 - start) * 1000)) + "ms")
                det_bboxes, det_conf, det_classid = post_process(outputs[0])
                print("延时1：" + str(int((time.perf_counter() - start1) * 1000)) + "ms")
                # result_img = img_draw(srcimg,det_bboxes, det_conf, det_classid,newh, neww, top, left)
                if len(det_bboxes.shape) > 1:
                    byte_result = np.concatenate([det_bboxes, det_conf.reshape((-1,1)), det_classid.reshape((-1,1))],axis=1).tostring()
                else:
                    byte_result = np.concatenate([np.array([[0,0,0,0]]), np.array([[2]]), np.array([[0]])],axis=1).tostring()
                # cv2.imshow("img", img)
                # cv2.waitKey(1)
                # cv2.imwrite(r'/home/toybrick/nanodet-service/img3_result.jpg',result_img)

                print(byte_result)
                # 发送推理结果，可以开始下一帧图像的传输
                client_socket.send(byte_result)
                print('yi fa song jie guo')

            else:
                print("已断开！")
                break
    finally:
        client_socket.close()
rknn.release()
