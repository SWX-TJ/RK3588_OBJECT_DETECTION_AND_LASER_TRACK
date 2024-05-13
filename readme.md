
## Project Description
This is an open-source laser control system. The tracking of the object is based on YOLOV8 and KCF. The control of the laser is based on a two-dimensional galvanometer system. The main control chip of the whole system is the RK3588s of Rockchip Technology and the STM32F103C8T6 of STMicroelectronics. You can train your own custom datasets for object detection and tracking. We achieved 14FPS real-time capability on the embedded board.
## Software and Hardware Environment
| Hardware      | Description |
| ----------- | ----------- |
| RK3588s      | 8 core ARM A76+A55 6TOPS|
| STM32F103C8T6   | ARM cortex-M3 |

### RK3588s Resource
![rk3588](https://www.scensmart.com/wp-content/uploads/2021/12/RK3588S.png)


| Software Requirments     
| -----------
|Ubuntu 20.04 
python==3.8.10    
cvui==2.7
matplotlib==3.1.2
numpy==1.24.1
argparse==1.4.0
scikit-learn==1.3.2 
opencv_contrib_python==4.9.0.80
opencv_python==4.9.0.80
pyserial==3.5
rknn_toolkit_lite2==1.6.0

## 配置
### 系统配置
1. linux下串口需要分配权限，可以在终端输入如下指令：
```markdown
sudo chmod 777 /dev/ttyS0 #/dev/ttyS0对应的是linux下的串口号，可以根据自己设置的串口来修改。
``` 
2. 也可以将上面指令放在~/.bashrc里面作为系统开机自启项目
```markdown
sudo vim ~/.bashrc #在打开的文档最后添加上述指令后保存并退出
source #退出后输入该指令或者重启系统以生效
```
每次换一个新的纯净系统只需要配置一次，无需每次都配置。



## 标定
### 双目相机参数标定
相机内参这里我们使用的是张正友标定算法，相机外参标定我们使用的是立体标定与立体校正。预先准备：一个棋盘格标定板。棋盘格子的长和宽需要已知。标定步骤如下：
1. 运行相机标定数据采集程序：
```markdown
./your_project_dir/dist/GetCameraCalibration --camera_type 0 --width 640 --height 480
```
注意：
* 注意your_project_dir是你的项目所在的目录
* 注意程序里默认的相机类型：camera_type，0表示相机输出为一幅图像，如果选择的相机输出的是左右两幅图像，则需要等作者更新相应的程序后再标定。
* --width 640 --height 480 则分别表示左右相机一幅图的长宽，可以根据购买的设备设置。
* 棋盘格标定板每次都要能出现在左右两个视图里
* 棋盘格随意变换角度、距离后再点击一次s按键
* 棋盘格距离最好覆盖不同的深度，保证采集的图像>=60对。
2. 采集后的图像保存在cameracalibrationdatasets文件夹下面。该文件夹的结构如下图所示：
```
cameracalibrationdatasets
├── left
│   ├── left_xxx.jpg
│   └── left_xxx.jpg
└── right
    ├── right_xxx.jpg
    └── right_xxx.jpg
```
3. 下载该文件夹并压缩后命名成chess_width.zip发给作者,其中width为标定板中一个格子的物理宽度，以mm为单位，比如27mm,可以命名为chess_27.zip，作者标定后会返回对应的相机参数文件camera.xml。
4. 用户如果自己会相机内参、外参标定，也可以自行标定，只要最后的相机参数文件格式和用户的一致即可。
### 立体匹配参数设置
我们使用的是BM算法做立体匹配，BM算法需要调整参数，参数调整可以参考google其他人的调整方法。调整程序如下：
```
./dist/StereoBMParamSetting --camera_param_path "camera.xml" --width 640 --height 480
```
通过拖动滑动条调整参数，鼠标左键点击带滚动条的画面，如果点击的点测距结果准确，则可以点击按键s保存参数。参数文件为stereobm.json.

### 激光振镜-相机联合标定
为了能够根据双目相机系统里的目标点在完成双目标定程序后，我们可以通过双目系统获取空间中的激光点的三维坐标，这里我们使用一个神经网络来拟合控制振镜偏转的DAC值和其激光的映射关系。神经网络结构比较简单，是一个三层MLP网络，输入是DAC值，输出是激光点的三维坐标。然后我们通过求解argmin问题，可以反求要到达要求的激光点的目标值时所需要的DAC输入。
具体操作：
1. 首先运行数据集采集制作程序
```
./dist/GetLaserMirrorCalibration --sample_nums 10 --pointmatrix_rows 20 --pointmatrix_cols 20
```
* 程序运行后，首先设置Binary窗口里的滑动条调整灰度，使得画面里基本上只有激光点
* 点击按键s，标定自动启动，程序会自行采集激光点在双目相机下的三维坐标
* 一次点阵标定完成后，终端会提示点击按键m。此时先在z轴方向上移动标定板，再点击按键m,此时会显示跟踪的完整点阵的画面。
* 当设置的采样次数达到后，程序提示sampled finished,然后保存数据集文件名为total_datasets.npy,此时按下按键q退出即可。
2. 离线标定（非公开）
采集得到的数据集提交给作者，作者会标定后给出激光振镜系统和双目系统之间的出厂参数文件lasercamera.npz.
3. 在线标定微调 (TBD)
用户的设备由于运输或者搬移过程中发生相机和激光的相对位置变化、镜头参数变化，这时候需要进行在线标定用于微调结果。用户只需要运行如下程序进行微调即可。
```
等待作者更新
```
## 目标检测算法训练及部署
本项目中我们使用的是Yolo算法用于目标检测。关于Yolo算法的详细解释自行google。我们这里简要说明如何训练自己的数据集。
### 训练数据集准备
自定义数据集参考这篇CSDN博客。
https://blog.csdn.net/fjlaym/article/details/123992962?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168722415116800180654496%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168722415116800180654496&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-123992962-null-null.142%5Ev88%5Econtrol,239%5Ev2%5Einsert_chatgpt&utm_term=yolov5%E5%88%B6%E4%BD%9C%E8%87%AA%E5%B7%B1%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86&spm=1018.2226.3001.4187
### 模型训练及结果保存
从瑞芯微官网下载Yolov8的源代码。
```
git clone https://github.com/airockchip/ultralytics_yolov8.git
```
下载源码完成后，进入文件夹，然后运行：
```
python install -e .
```
安装完成后，在终端里输入yolo，如果执行不报错，则说明安装成功。
利用第一步制作的数据集，参考源码里的训练步骤进行训练。训练完成后使用如下代码导出onnx模型：
```
yolo mode=export model=best.pt format=rknn
```
### 模型在RK3588上部署
首先要保证安装了rknn-toolkit2，这个安装请自行百度。
然后下载rknn_model_zoo，这里注意版本要和rknn-toolkit的版本保持一致。下载完成后进入your project path/rknn_model_zoo-1.6.0/examples/yolov8/python目录下。运行如下命令转换onnx为rknn模型。
```
python convert.py your project path/xxxyourmodel.onnx rk3588 i8 xxx.rknn
```
将转换好的模型下载到板子上，即可接入主程序。
## 程序运行
本程序采用线程池加速运行，在代码中可以修改线程池中线程个数，可以修改参数文件的真实地址。一切都修改就绪后，通过运行如下程序即可实现自动目标检测和激光跟随。
```
python multiprocess_main.py
```

## 主要贡献者
| 姓名      | 任务分工 |
| ----------- | ----------- |
| 沈文祥      | 机器视觉+人工智能+linux系统开发|
| 张金时   | 嵌入式微控制器开发 |
| 董力纲   | 电路设计+机械制作 |
