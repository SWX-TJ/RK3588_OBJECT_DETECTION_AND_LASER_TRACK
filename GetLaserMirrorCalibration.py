import serial
import time
import numpy as np
import cv2
import argparse
from sklearn.decomposition import PCA
from multiprocessing import Process, Queue
from dist.cameraParams import StereoCameraParams #双目相机内外参类
from stereoMatcherCls import StereoMatcher#双目匹配类

def PreprocessImage(oriimage,thr):
    if oriimage.ndim==3:
        gray_image = cv2.cvtColor(oriimage,cv2.COLOR_BGR2GRAY)
    else:
        gray_image = oriimage.copy()
    ret, binary = cv2.threshold(gray_image, thr, 255, cv2.THRESH_BINARY)
    return binary

def ImageFilterandFindCounter(ori_image):
    # 应用Canny对二值化图像进行边缘检测
    # 50 和 150 是Canny函数的低阈值和高阈值
    edges = cv2.Canny(ori_image, 0, 255)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_dicts = {}
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        contour_points = [point[0] for point in contour]
        perimeter = cv2.arcLength(contour, True)
        contour_dicts[i] = {'area':area,'perimeter':perimeter,'contour_points':np.array(contour_points)}
    return contour_dicts,edges


def find_median(lst):
    # 首先对列表进行排序
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    
    # 如果列表长度为奇数，中位数是中间的数
    if n % 2 != 0:
        return sorted_lst[n // 2]
    # 如果列表长度为偶数，中位数是中间两个数的平均值
    else:
        return (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2

#处理图像轮廓
def SelectContour(contour_dicts):
    proced_contour_dicts = {}
    area_lists = []
    perimeter_lists = []
    if len(contour_dicts)<1:
        return proced_contour_dicts
    for _, attributes in contour_dicts.items():
        area_lists.append(attributes['area'])
        perimeter_lists.append(attributes['perimeter'])
    #找中位数
    area_median = find_median(area_lists)
    #print("area_median",area_median)
    perimeter_median = find_median(perimeter_lists)
   # print("perimeter_median",perimeter_median)
    min_area_thr = area_median-area_median*0.5
    max_area_thr = area_median+area_median*0.5
    min_per_thr = perimeter_median-perimeter_median*0.5
    max_per_thr = perimeter_median+perimeter_median*0.5
    proc_contour_idx = 0
    for _, attributes in contour_dicts.items():
        if attributes['area']>min_area_thr and  attributes['area']<max_area_thr:
            if attributes['perimeter']>min_per_thr and  attributes['perimeter']<max_per_thr:
                proced_contour_dicts[proc_contour_idx] = {'area':attributes['area'],'perimeter':attributes['perimeter'],'contour_points':attributes['contour_points']}
                proc_contour_idx = proc_contour_idx+1
    return proced_contour_dicts


def FitContourEllipse(contour_dicts:dict):
    for contour_id, attributes in contour_dicts.items():
        contour_points = attributes['contour_points']
        (x, y), _ = cv2.minEnclosingCircle(contour_points)
        cx = int(x)
        cy = int(y)
        contour_dicts[contour_id]['center_pts'] = (cx,cy)
    return contour_dicts



def generate_dot_matrix(rows, cols):
    """
    根据指定的行数和列数，在[0, 65535]的范围内生成点阵。
    
    参数:
        rows (int): 行数
        cols (int): 列数
    返回:
        list of tuple: 生成的点阵，每个元素是一个包含(x, y)坐标的元组。
    """
    # 计算行和列的间隔
    start_pt = 1000
    end_pt = 65535
    row_spacing = (end_pt-start_pt) / (rows - 1) if rows > 1 else 0
    col_spacing =  (end_pt-start_pt)  / (cols - 1) if cols > 1 else 0
    
    # 生成点阵
    dot_matrix = []
    for i in range(rows):
        for j in range(cols):
            x = start_pt+round(j * col_spacing)
            y = start_pt+round(i * row_spacing)
            dot_matrix.append((x, y))
    return dot_matrix

def fit_3d_line(points):
    # 计算所有点的中心
    center = points.mean(axis=0)

    # 使用PCA找到数据的主方向
    pca = PCA(n_components=1)
    pca.fit(points)
    direction = pca.components_[0]
    return direction,center


def ProcOriDatasetsAndSaved(pointmatrix,total_batch_datasets):
    saved_datasets = []
    for point_dac in pointmatrix:
        dac_x,dac_y = point_dac
        cur_line_pts = []
        for idx in range(len(total_batch_datasets)):
                batch_datasets = total_batch_datasets[idx]
                for contour_id, attributes in batch_datasets.items():
                    if attributes['dac_x']==dac_x and attributes['dac_y']==dac_y:
                        line_pt = [attributes['wx'],attributes['wy'],attributes['wz']]
                        cur_line_pts.append(line_pt)
        cur_line_pts_np = np.array(cur_line_pts)
        direction,center = fit_3d_line(cur_line_pts_np)
        single_final_data = [dac_x,dac_y,direction[0],direction[1],direction[2],center[0],center[1],center[2]]
        saved_datasets.append(single_final_data)
    saved_datasets = np.array(saved_datasets)
    np.save(f"total_datasets_{len(total_batch_datasets)}.npy",saved_datasets)



# def preprocess(img1, img2):
#     def adjust_contrast(image, alpha):
#         """
#         调整图像的对比度
#         :param image: 原始图像
#         :param alpha: 对比度控制因子（大于1增加对比度，小于1但大于0减少对比度）
#         :return: 调整对比度后的图像
#         """
#         # 新图像 = alpha * 原图像 + beta
#         # 在这里 beta 设置为0，因为我们只调整对比度
#         new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
#         return new_image
#     # # 彩色图->灰度图
#     # if(img1.ndim == 3):#判断为三维数组
#     #     gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
#     # if(img2.ndim == 3):
#     #     gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     denoised_img1 = cv2.GaussianBlur(img1, (5, 5), 0)
#     denoised_img2 = cv2.GaussianBlur(img2, (5, 5), 0)
#     equalized_img1 = adjust_contrast(denoised_img1,1.2)#cv2.equalizeHist(denoised_img1)
#     equalized_img2 = adjust_contrast(denoised_img2,1.2) #cv2.equalizeHist(denoised_img2)
#     return equalized_img1, equalized_img2


def Serial_Process(control_data_from_image_queue,control_data_to_image_queue,data_from_image_queue,data_to_image_queue,port='/dev/ttyS0',baudrate=115200):
    while True:
        try:
            # 尝试建立串口连接
            with serial.Serial(port, baudrate, timeout=1) as ser:
                print(f"Connected to {port}.")
                # 当串口连接时
                while True:
                    if not ser.is_open:
                        break  # 如果串口关闭，退出内层循环重新连接
                    if not control_data_from_image_queue.empty():
                        data =control_data_from_image_queue.get()
                        dis_value = format(int(data[1]), '04X')
                        dis_value_default = format(0, '04X')
                        delay_time = float(data[1])*0.2
                        if data[0]==0: #拉回最近处
                            dir_value = format(29, '02X')
                            delay_time = 5#60#60*3#60*4 #假设是4分钟
                        elif data[0]==1: #推远最远处
                            dir_value = format(45, '02X')
                            delay_time = 60*4 #假设是4分钟
                        elif data[0]==2:#拉回指定距离
                            dir_value = format(17, '02X')
                        elif data[0]==3:#推远指定距离
                            dir_value = format(34, '02X')
                        # 将数据发送到串口
                        frame = f"aa55{dir_value}{dis_value_default}{dis_value}ea60"
                        print("Frame",frame)
                        ser.write(bytes.fromhex(frame))
                        time.sleep(delay_time)
                        control_data_to_image_queue.put(True)
                    if not data_from_image_queue.empty():
                        # 从队列中获取数据
                        data = data_from_image_queue.get()
                        thx = format(int(data[0]), '04X')
                        thy = format(int(data[1]), '04X')  
                        # 将数据发送到串口
                        frame = f"aa55a0{thx}{thy}ea60"
                        print("Laser",frame)
                        ser.write(bytes.fromhex(frame))
                        #延时15ms等待移动到位发送给图像程序进行识别
                        time.sleep(0.21)
                        data_to_image_queue.put(True)
        except serial.SerialException:
            print(f"Connection to {port} failed. Retrying...")
            time.sleep(5)  # 等待一段时间后重试连接
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get LaserMirror TrainDatasets')
    # 这里的描述会出现在 usage下方 表明这个程序的作用是什么
    parser.add_argument("--serialport", type=str, default="/dev/ttyS0")
    parser.add_argument("--serialbaund", type=int, default=115200)
    parser.add_argument("--camera_param_filepath", type=str, default='./camera_0412.xml')
    parser.add_argument("--stereobm_filepath", type=str, default='./stereosgbm.json')
    parser.add_argument("--pointmatrix_rows", type=int, default=30)
    parser.add_argument("--pointmatrix_cols", type=int, default=30)
    parser.add_argument("--sample_nums", type=int, default=20)
    parser.add_argument("--max_distance_thr", type=int, default=300)
    parser.add_argument("--min_distance_thr", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    print("CameraParamPath: ",args.camera_param_filepath)
    print("StereoBMPath: ",args.stereobm_filepath)
    print("SerialPort: ",args.serialport)
    print("Serialbaund: ",args.serialbaund)
    print("PointMatrix_Rows: ",args.pointmatrix_rows)
    print("PointMatrix_Cols: ",args.pointmatrix_cols)
    print("TotalSample Nums: ",args.sample_nums)
    print("Max Distance Threshold: ",args.max_distance_thr)
    print("Min Distance Threshold: ",args.min_distance_thr)
    print("Image Width: ",args.width)
    print("Image Height: ",args.height)
    #初始化串口
    data_from_image_queue = Queue()
    data_to_image_queue = Queue()
    control_data_from_image_queue = Queue()
    control_data_to_image_queue = Queue()
    serial_proc = Process(target=Serial_Process, args=(control_data_from_image_queue,control_data_to_image_queue,data_from_image_queue,data_to_image_queue,args.serialport,args.serialbaund,))
    serial_proc.start()
    #默认先回到最近点
    data = [0,0]
    control_data_from_image_queue.put(data)
    #等待延迟结束
    while 1:
        if not control_data_to_image_queue.empty():
            break
    data = [3,80]#标定原点
    control_data_from_image_queue.put(data)
    #等待延迟结束
    while 1:
        if not control_data_to_image_queue.empty():
            break
    #1.加载相机参数
    camera_config = StereoCameraParams(args.camera_param_filepath)
    cap = cv2.VideoCapture(0+cv2.CAP_V4L2)
    # 设置USB双目摄像头分辨率
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.image_width*2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.image_height)
    cap.set(cv2.CAP_PROP_FPS,30)
    #设置双目立体匹配器
    m_stereomatcher = StereoMatcher(camera_config,args.stereobm_filepath,'SGBM')
    #生成待标定点阵坐标
    rows_m = args.pointmatrix_rows
    cols_m = args.pointmatrix_cols
    m_pointmatrix = generate_dot_matrix(rows_m,cols_m)
    pointmatrix_idx = 0
    single_image_point_info = {}
    total_datasets = []
    cv2.namedWindow("Binary")
    cv2.createTrackbar("threshold", "Binary", 230, 255, lambda x: None)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            #1.get orign image
            iml  = frame[:, 0:camera_config.image_width, :]
            imr  = frame[:, camera_config.image_width:camera_config.image_width*2, :]
            #2.cacu stereo
            #gray_iml,gray_imr = preprocess(iml,imr) 
            depthImage,iml_rectified_l,disparity_color = m_stereomatcher.CacuDepth(iml,imr,True)
            def callbackFunc(e, x, y, f, p):
                if e == cv2.EVENT_LBUTTONDOWN:        
                    #print('点 (%d, %d) 的三维坐标 (%fmm, %fmm, %fmm)' % (x, y, depthImage[y, x, 0], depthImage[y, x, 1], depthImage[y, x, 2]))
                    distance  = ( (depthImage[y, x, 0] ** 2 + depthImage[y, x, 1] ** 2 + depthImage[y, x, 2] **2) ** 0.5) / 10
                    print('点 (%d, %d) 距离左摄像头的相对距离为 %0.3f cm, %0.3f cm' %(x, y,  distance,abs(depthImage[y, x,2])/10.0) )
            binary_thr = cv2.getTrackbarPos("threshold", "Binary")
            binary_image = PreprocessImage(iml_rectified_l,binary_thr)
            if not data_to_image_queue.empty():
                #串口线程完成发送
                result = data_to_image_queue.get()
                #再刷新5帧保证收到的图像是稳定的
                for i in range(3):
                    while (1):
                        ret, frame = cap.read()
                        if ret:
                            #1.get orign image
                            iml  = frame[:, 0:camera_config.image_width, :]
                            imr  = frame[:, camera_config.image_width:camera_config.image_width*2, :]
                            #2.cacu stereo 
                            #gray_iml,gray_imr = preprocess(iml,imr) 
                            depthImage,iml_rectified_l,disparity_color = m_stereomatcher.CacuDepth(iml,imr,True)
                            binary_image = PreprocessImage(iml_rectified_l,binary_thr)
                            break
                #轮廓识别
                contour_dicts,merged_edge_images= ImageFilterandFindCounter(binary_image)
                proced_contour_dicts = SelectContour(contour_dicts)
                if len(proced_contour_dicts)==1:
                    #拟合最小包围圆
                    proced_contour_dicts = FitContourEllipse(proced_contour_dicts)
                    cur_cx,cur_cy = proced_contour_dicts[0]['center_pts']
                    world_center_pts = (depthImage[cur_cy, cur_cx, 0]/10.0, depthImage[cur_cy, cur_cx, 1]/10.0, depthImage[cur_cy, cur_cx, 2]/10.0)
                    #rint("world_center_pts",world_center_pts)
                    if world_center_pts[2]>args.min_distance_thr and  world_center_pts[2]<args.max_distance_thr:
                        dac_x,dac_y = m_pointmatrix[pointmatrix_idx]
                        #print("world_center_pts",world_center_pts)
                        single_image_point_info[pointmatrix_idx] = {'dac_x':dac_x,'dac_y':dac_y,'cx':cur_cx,'cy':cur_cy,'wx':world_center_pts[0],'wy':world_center_pts[1],'wz':world_center_pts[2]}
                    pointmatrix_idx+=1
                if pointmatrix_idx<rows_m*cols_m:
                    print("Current Point ID: ",pointmatrix_idx,m_pointmatrix[pointmatrix_idx-1])
                    data_from_image_queue.put(m_pointmatrix[pointmatrix_idx])
                else:
                    if len(single_image_point_info)!=rows_m*cols_m:
                        print("Auto Search Laser Point Again")
                        pointmatrix_idx = 0
                        data_from_image_queue.put(m_pointmatrix[pointmatrix_idx])
                    else:
                        #留着用于发送控制滑轨自动移动,如果没有自动移动就使用m按键控制
                        print("Point Matrix Info Detect finished, Press m to saved current point matrix info!")
                        print("Detect Total Point Nums: ",len(single_image_point_info))
                        single_image = iml_rectified_l.copy()
                        for contour_id, attributes in single_image_point_info.items():
                            cx,cy = attributes["cx"],attributes["cy"]
                            cv2.circle(single_image,(cx,cy),1,(0,255,0),2)
                        #保存当前距离下的数据
                        if len(total_datasets)<args.sample_nums:
                            total_datasets.append(single_image_point_info)
                            single_image_point_info = {}
                            if len(total_datasets)>=3:
                                ProcOriDatasetsAndSaved(m_pointmatrix,total_datasets)
                            print(f"Sampled Progress---{len(total_datasets)}/{args.sample_nums}")
                            #移动距离
                            control_data = [3,10] #每次移动1cm
                            control_data_from_image_queue.put(control_data)
                            while 1:
                                if not control_data_to_image_queue.empty():
                                    break
                            pointmatrix_idx = 0
                            data_from_image_queue.put(m_pointmatrix[pointmatrix_idx])
                        else:
                            ProcOriDatasetsAndSaved(m_pointmatrix,total_datasets)
                            total_datasets = []
                            single_image_point_info = {}
                            print("Sampled Finished!")
                        cv2.imshow("PointMatrixDisp",single_image)
            cv2.imshow("Color",iml_rectified_l)
            cv2.setMouseCallback("Color", callbackFunc, None)
            cv2.imshow("Binary",binary_image)
            cv2.imshow("Depth",disparity_color)
            key = cv2.waitKey(1)
            if key==ord("e"):
                merged_edge_images = None
                send_matrix_idx = 0
                print("Start Moved Laser and Detect Point \n")
                print("Current Point ID: ",send_matrix_idx)
                data_from_image_queue.put(m_pointmatrix[send_matrix_idx])
            if key==ord("w"):
                #移动距离
                control_data = [3,50] #每次移动1cm
                control_data_from_image_queue.put(control_data)
                while 1:
                    if not control_data_to_image_queue.empty():
                        break
            if key==ord("s"):
                #移动距离
                control_data = [2,50] #每次移动1cm
                control_data_from_image_queue.put(control_data)
                while 1:
                    if not control_data_to_image_queue.empty():
                        break
            # if key == ord("m"):
            #     print("Detect Total Point Nums: ",len(single_image_point_info))
            #     single_image = iml_rectified_l.copy()
            #     for contour_id, attributes in single_image_point_info.items():
            #         cx,cy = attributes["cx"],attributes["cy"]
            #         cv2.circle(single_image,(cx,cy),1,(0,255,0),2)
            #     #保存当前距离下的数据
            #     if len(total_datasets)<args.sample_nums:
            #         total_datasets.append(single_image_point_info)
            #         single_image_point_info = {}
            #         print(f"Sampled Progress---{len(total_datasets)}/{args.sample_nums}")
            #     else:
            #         ProcOriDatasetsAndSaved(m_pointmatrix,total_datasets)
            #         total_datasets = []
            #         single_image_point_info = {}
            #         print("Sampled Finished!")
            #     cv2.imshow("PointMatrixDisp",single_image)
            if key == ord("q"):
                  break
    cap.release()
    cv2.destroyAllWindows()
    serial_proc.terminate()
    print('Stop Exe Sucessed!')
    serial_proc.join()
            
            
                    
                
    
    
    

