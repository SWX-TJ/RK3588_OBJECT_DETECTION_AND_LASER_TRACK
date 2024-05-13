import cv2
import cvui
import time
import numpy as np
import serial
from multiprocessing import Process, Queue
from MultiTaskPool import TaskPoolExecutor
from dist.cameraParams import StereoCameraParams #双目相机内外参类
from dist.singleObjTracker import Tracker #单目标跟踪类
from dist.slfn_layers import SLFNModel
from skimage import restoration
from yolov8 import YoloObjectDetectionandMeasureDistance,justdrawdetectionresult,selecttrackedobjinfo

# def deblur(img):
#     blurred_image = img / 255.0  # 归一化到0-1，如果图像未归一化
#     def gaussian_kernel(size, sigma):
#         """生成高斯模糊核"""
#         kernel = np.zeros((size, size), dtype=np.float32)
#         for x in range(-size//2, size//2 + 1):
#             for y in range(-size//2, size//2 + 1):
#                 kernel[x + size//2, y + size//2] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
#         return kernel / kernel.sum()
#     psf = gaussian_kernel(3, 1)  # 假设高斯核大小为5x5，σ=1
#     # 对每个颜色通道应用Richardson-Lucy去卷积
#     deblurred_channels = []
#     for channel in range(3):
#         deblurred = restoration.richardson_lucy(blurred_image[:, :, channel], psf, num_iter=10)
#         deblurred_channels.append(deblurred)
#     # 将处理后的通道重新组合成一个彩色图像
#     deblurred_image = np.stack(deblurred_channels, axis=-1)
#     print("deblurred_image",deblurred_image.shape)
#     # 将去模糊后的图像数据转换为0到255的范围并转换为uint8类型
#     final_img = (deblurred_image * 255).astype(np.uint8)
#     return final_img

def preprocess(img1, img2):
    # def adjust_contrast(image, alpha):
    #     """
    #     调整图像的对比度
    #     :param image: 原始图像
    #     :param alpha: 对比度控制因子（大于1增加对比度，小于1但大于0减少对比度）
    #     :return: 调整对比度后的图像
    #     """
    #     # 新图像 = alpha * 原图像 + beta
    #     # 在这里 beta 设置为0，因为我们只调整对比度
    #     new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    #     return new_image
    proc_img1 = cv2.GaussianBlur(img1, (3, 3),0)
    proc_img2 = cv2.GaussianBlur(img2, (3, 3),0)
    # proc_img1 = deblur(img1)
    # proc_img2 = deblur(img2)

    return proc_img1,proc_img2

#用户可以自己的参数位置根据具体位置修改
CAMERA_XML_FILEPATH="camera_0412.xml"
STEREOBM_JSON_FILEPATH = "stereobm_0412.json"
CAMERA_LASER_PARAM_FILEPATH = "model_param.npz"
RKNN_YOLOV8_FILEPATH="rknn_model/mbest.rknn"#"rknn_model/yolov8.rknn"#"rknn_model/mbest.rknn"
RKNN_YOLOV8_CLASS_FILEPATH = "rknn_model/classes.txt"
RKNN_POOL_NUMS = 8 #线程池大小
SCREEN_SIZE = (1024,600)# 7-in screen size



def serial_process(send_queue, port='/dev/ttyS0', baudrate=115200):
    while True:
        try:
            # 尝试建立串口连接
            with serial.Serial(port, baudrate, timeout=1) as ser:
                print(f"Connected to {port}.")
                # 当串口连接时
                while True:
                    if not ser.is_open:
                        break  # 如果串口关闭，退出内层循环重新连接
                    if not send_queue.empty():
                        # 从队列中获取数据
                        data = send_queue.get()
                        dac_x,dac_y = data
                        if dac_x>=65535:
                            dac_x = 65535
                        elif dac_x<=0:
                            dac_x = 0
                        if dac_y >=65535:
                            dac_y = 65535
                        elif dac_y<=0:
                            dac_y = 0
                        thx = format(int(dac_x), '04X')
                        thy = format(int(dac_y), '04X')
                        # 将数据发送到串口
                        frame = f"aa55a0{thx}{thy}ea60"
                        ser.write(bytes.fromhex(frame))  
        except serial.SerialException:
            print(f"Connection to {port} failed. Retrying...")
            time.sleep(5)  # 等待一段时间后重试连接



# ObjClassNameList = ["person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
#            "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
#            "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
#            "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
#            "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
#            "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
#            "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush "]

if __name__ == '__main__':
    #0.初始化神经网络
    slfnmodel = SLFNModel(n_hidden=165,layer_params_filepath=CAMERA_LASER_PARAM_FILEPATH)
    #加载目标名称的List
    ObjClassNameList = []
    with open(RKNN_YOLOV8_CLASS_FILEPATH, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 去除每行末尾的换行符
    ObjClassNameList = [line.strip() for line in lines]
    with open(RKNN_YOLOV8_CLASS_FILEPATH, "r") as f:
        class_name = f.readline()
        ObjClassNameList.append(class_name)
    print("Class Name List Num -->",len(ObjClassNameList))
    #1.加载相机参数
    camera_config = StereoCameraParams(CAMERA_XML_FILEPATH)
    cap = cv2.VideoCapture(0+cv2.CAP_V4L2)
    # 设置USB双目摄像头分辨率
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.image_width*2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.image_height)
    cap.set(cv2.CAP_PROP_FPS,30)
    #初始化串口
    serial_queue = Queue()
    serial_proc = Process(target=serial_process, args=(serial_queue,))
    serial_proc.start()
    #初始化UI
    WINDOW_NAME = 'Object Detection'
    cvui.init(WINDOW_NAME)
    ui_frame = np.zeros((SCREEN_SIZE[1],SCREEN_SIZE[0], 3), np.uint8)
    clicked_object_cord = (-100,-100)
    is_quit_exe = False
    is_select_object_to_tracker = False
    def on_clickedleftmouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            global clicked_object_cord
            clicked_object_cord = (x,y)
            global is_select_object_to_tracker
            is_select_object_to_tracker = True
            if x>camera_config.image_width or y>camera_config.image_height:
                global is_quit_exe
                is_quit_exe = True
    cv2.setMouseCallback(WINDOW_NAME, on_clickedleftmouse)
    loopTime=0
    frames_id = 0
    averfpsvalue = 0
    #初始化跟踪器
    m_tracker = Tracker()
    #初始化图像处理多线程
    proc_pool = TaskPoolExecutor(rknnModel=RKNN_YOLOV8_FILEPATH,cameraParams=camera_config,stereobmParams=STEREOBM_JSON_FILEPATH,TPEs=RKNN_POOL_NUMS,func=YoloObjectDetectionandMeasureDistance)
    # 初始化异步所需要的帧
    if (cap.isOpened()):
        for i in range(RKNN_POOL_NUMS + 1):
            ret, frame = cap.read()
            iml  = frame[:, 0:camera_config.image_width, :]
            imr  = frame[:, camera_config.image_width:camera_config.image_width*2, :]
            if not ret:
                cap.release()
                del proc_pool
                exit(-1)
            proc_pool.put(iml,imr) 
    while (cap.isOpened()):
        ret, frame = cap.read()
        ui_frame[:] = (0,0,0)
        if ret:
            #1.get orign image
            iml  = frame[:, 0:camera_config.image_width, :]
            imr  = frame[:, camera_config.image_width:camera_config.image_width*2, :]
            #preprocess
            proc_pool.put(iml,imr)
            results, flag = proc_pool.get()
            real_boxes, classes, scores,depthImage,img_p = results
            #.3 update ui
            cv2.rectangle(img_p, (100, 50), (490, 442), (0, 255, 0), 3)
            ui_frame[0:camera_config.image_height, 0:camera_config.image_width] = img_p
            if real_boxes is not None: #如果有物体检测则判断用户是否需要跟踪某个物体
                if is_select_object_to_tracker:  #用户需要跟踪物体
                    is_select_object_to_tracker = False
                    #1.获取需要跟踪的物体的包围框和类别
                    image,tracked_bbox,tracked_world_pos,tracked_object_cls=selecttrackedobjinfo(img_p,real_boxes,scores, classes,depthImage,clicked_object_cord,ObjClassNameList)
                    #2.初始化跟踪器
                    if tracked_bbox[0] !=-1 and m_tracker.isWorking==False: #初始化新的跟踪器
                        m_tracker.inittracker(img_p,tracked_bbox)
                    else:
                        m_tracker.isWorking = False
                    ui_frame[0:camera_config.image_height, 0:camera_config.image_width] = image
                else:#用户没有任何点击需要跟踪物体
                    if m_tracker.isWorking: #上一个跟踪器还在运行
                        image,real_cords = m_tracker.track_step(img_p,depthImage)
                        if real_cords[0]!=-10000:
                            object_info_str = f"Object Name: {tracked_object_cls}"
                            cvui.text(ui_frame,camera_config.image_width+10, 70, object_info_str,0.8,0x00FF00)
                            object_cords_str = f"Object Cords: {int(real_cords[0])},{int(real_cords[1])},{int(real_cords[2])}"
                            cvui.text(ui_frame,camera_config.image_width+10, 100, object_cords_str,0.8,0x00FF00)
                            pred_dac = slfnmodel.predict_best_dac(real_cords)    
                            serial_queue.put(pred_dac)  # 放入队列
                            #发送数据到串口进程
                        cv2.rectangle(image, (100, 50), (490, 442), (0, 255, 0), 3)
                        ui_frame[0:camera_config.image_height, 0:camera_config.image_width] = image
                    else:  #没有任何跟踪器，只显示检测结果
                        justdrawdetectionresult(img_p,real_boxes,scores,classes,depthImage,ObjClassNameList)
                        cv2.rectangle(img_p, (100, 50), (490, 442), (0, 255, 0), 3)
                        ui_frame[0:camera_config.image_height, 0:camera_config.image_width] = img_p
        if frames_id !=0 and frames_id % 10 == 0:
            averfpsvalue = 10/ (time.time() - loopTime)
            frames_id = 0
            loopTime = time.time()
        fps_info_str = 'FPS: %.1f'%(averfpsvalue)
        cvui.text(ui_frame,camera_config.image_width+10, 10,fps_info_str,0.8,0x00FF00)
        cvui.text(ui_frame,camera_config.image_width+10, 40, "Track Object Info",0.8,0x00FF00)
        cvui.update()
        cvui.imshow(WINDOW_NAME, ui_frame)
        frames_id+=1
        key = cv2.waitKey(1)
        if is_quit_exe: #退出主程序
            break
    cap.release()
    cv2.destroyAllWindows()
    proc_pool.release()
    serial_proc.terminate()
    serial_proc.join()
