from dist.cameraParams import StereoCameraParams #双目相机内外参类
from stereoMatcherCls import StereoMatcher#双目匹配类
from dist.slfn_layers import offline_train,SLFNModel
from multiprocessing import Process, Queue
import numpy as np
import cv2
import time
import serial



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
                        dac_x,dac_y = data#findBestDac(data)
                        print("Get DAC Value-->",dac_x,dac_y)
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


# if __name__ == "__main__":

#     slfnmodel = SLFNModel(n_hidden=165,layer_params_filepath="./model_param.npz")
#     target_pt = np.array([4.661379, -0.963149, 99.011707])#
#     dac_value = slfnmodel.predict_best_dac(target_pt)    
#     print("dac_value",dac_value)
if __name__ == "__main__":
    # traindatasets = np.load("total_datasets_14.npy")
    # X_train = traindatasets[:,:2]
    # Y_train = traindatasets[:,2:]
    # offline_train(X_train,Y_train)
    slfnmodel = SLFNModel(n_hidden=165,layer_params_filepath="./model_param.npz")
    #1.加载相机参数
    camera_config = StereoCameraParams("./camera_0412.xml")
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
    m_stereomatcher = StereoMatcher(camera_config,'./stereosgbm.json','SGBM')
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            #1.get orign image
            iml  = frame[:, 0:camera_config.image_width, :]
            imr  = frame[:, camera_config.image_width:camera_config.image_width*2, :]
            #2.cacu stereo 
            depthImage,iml_rectified_l,disparity_color = m_stereomatcher.CacuDepth(iml,imr,True)
            def callbackFunc(e, x, y, f, p):
                if e == cv2.EVENT_LBUTTONDOWN:
                    world_pts = [depthImage[y, x, 0]/10.0, depthImage[y, x, 1]/10.0, depthImage[y, x, 2]/10.0]
                    distance  = ( (depthImage[y, x, 0] ** 2 + depthImage[y, x, 1] ** 2 + depthImage[y, x, 2] **2) ** 0.5) / 10
                    print('点 (%d, %d) 的三维坐标 (%f, %f, %f)' % (x, y, depthImage[y, x, 0]/10.0, depthImage[y, x, 1]/10.0, depthImage[y, x, 2]/10.0))
                    if distance>10 and distance<300:
                        pred_dac = slfnmodel.predict_best_dac(world_pts)    
                        serial_queue.put(pred_dac)  # 放入队列
            #cv2.rectangle(iml_rectified_l, (100, 50), (490, 442), (0, 255, 0), 1)
            cv2.imshow("color",iml_rectified_l)
            cv2.imshow("depth",disparity_color)
            cv2.setMouseCallback("color", callbackFunc, None)
            key = cv2.waitKey(1)
            if key == ord("s"):
                cv2.imwrite("test.jpg",iml_rectified_l)
            if key == ord("q"):
                  break
    cap.release()
    cv2.destroyAllWindows()
    serial_proc.terminate()
    print('Stop Exe Sucessed!')
    serial_proc.join()
    