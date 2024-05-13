import cv2
import numpy as np
import json
import argparse
import shutil
import os
from dist.cameraParams import StereoCameraParams

def preprocess(img1, img2):
    def adjust_contrast(image, alpha):
        """
        调整图像的对比度
        :param image: 原始图像
        :param alpha: 对比度控制因子（大于1增加对比度，小于1但大于0减少对比度）
        :return: 调整对比度后的图像
        """
        # 新图像 = alpha * 原图像 + beta
        # 在这里 beta 设置为0，因为我们只调整对比度
        new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        return new_image
    # # 彩色图->灰度图
    if(img1.ndim == 3):#判断为三维数组
        gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if(img2.ndim == 3):
        gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    denoised_img1 = cv2.GaussianBlur(gray_img1, (5, 5), 0)
    denoised_img2 = cv2.GaussianBlur(gray_img2, (5, 5), 0)
    equalized_img1 =cv2.equalizeHist(denoised_img1) #adjust_contrast(denoised_img1,1.2)#cv2.equalizeHist(denoised_img1)
    equalized_img2 =cv2.equalizeHist(denoised_img2) #adjust_contrast(denoised_img2,1.2) #cv2.equalizeHist(denoised_img2)
    return equalized_img1, equalized_img2

# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)
    return undistortion_image

#获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T
    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, 
                                                    (width, height), R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
    return map1x, map1y, map2x, map2y, Q, roi1, roi2


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    return rectifyed_img1, rectifyed_img2

# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2
    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    return output



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stereo Camera Params')
    # 这里的描述会出现在 usage下方 表明这个程序的作用是什么
    parser.add_argument("--camera_param_path", type=str, default="camera_0412.xml")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    print("Camera Param FilePath: ",args.camera_param_path)
    print("Image Width: ",args.width)
    print("Image Height: ",args.height)
    print("##################### Stereo Camera Param Setting#####################")
    print("##################### Company: SHANGHAI JICHI Tech#####################")
    print("##################### Version: V1.0#####################")
    print("##################### Date: 2024.04.05#####################")
    print("\r\n")
    print("##################### Start Stereo Camera Param Setting #####################")
    print("##################### Press Key: s to saved Params #####################")
    print("##################### Press Key: q to quit Exe #####################")
    saved_json_filepath = "stereobm.json"
    if os.path.exists(saved_json_filepath):
          os.remove(saved_json_filepath)
    #加载相机参数
    xml_file = args.camera_param_path
    camera_config = StereoCameraParams(xml_file)
    cap = cv2.VideoCapture(0+cv2.CAP_V4L2)
    # 设置摄像头分辨率
    cap.set(6,cv2.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width*2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cv2.namedWindow("left")
    cv2.namedWindow("right")
    cv2.namedWindow("depth")
    cv2.moveWindow("left", 0, 0)
    cv2.moveWindow("right", 600, 0)
    
    cv2.createTrackbar("num", "depth", 2, 10, lambda x: None)
    cv2.createTrackbar("blockSize", "depth", 33, 255, lambda x: None)
    cv2.createTrackbar("prefilp", "depth", 31, 100, lambda x: None)
    cv2.createTrackbar("ttthread", "depth", 2, 100, lambda x: None)
    cv2.createTrackbar("uniqRatio", "depth", 1, 100, lambda x: None)
    cv2.createTrackbar("swsize", "depth", 100, 100, lambda x: None)
    cv2.createTrackbar("sprange", "depth", 32, 100, lambda x: None)
    cv2.createTrackbar("mind", "depth", 0, 20, lambda x: None)
    cv2.createTrackbar("dispmd", "depth", 1, 20, lambda x: None)
    map1x, map1y, map2x, map2y, Q,validPixROI1, validPixROI2 = getRectifyTransform(camera_config.image_height, camera_config.image_width, camera_config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
    stereo = cv2.StereoBM_create(numDisparities=160, blockSize=43)
    stereo.setROI1(validPixROI1)
    stereo.setROI2(validPixROI2)
    stereo.setPreFilterCap(21)
    stereo.setMinDisparity(0)
    stereo.setUniquenessRatio(2)
    stereo.setTextureThreshold(10)
    stereo.setSpeckleWindowSize(100)
    stereo.setSpeckleRange(16)
    stereo.setDisp12MaxDiff(1)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            #1.get orign image
            iml  = frame[:, 0:args.width, :]
            imr  = frame[:, args.width:args.width*2, :]
            # 消除畸变
            iml_und = undistortion(iml, camera_config.cam_matrix_left, camera_config.distortion_l)
            imr_und = undistortion(imr, camera_config.cam_matrix_right, camera_config.distortion_r)
            iml_, imr_ = preprocess(iml, imr)
            iml_rectified_l, imr_rectified_r = rectifyImage(iml_, imr_, map1x, map1y, map2x, map2y)
            # 两个trackbar用来调节不同的参数查看效果
            num = cv2.getTrackbarPos("num", "depth")
            blockSize = cv2.getTrackbarPos("blockSize", "depth")
            if blockSize % 2 == 0:
                blockSize += 1
            if blockSize < 5:
                blockSize = 5
            prefilp = cv2.getTrackbarPos("prefilp", "depth")
            ttthread = cv2.getTrackbarPos("ttthread", "depth")
            uniqRatio = cv2.getTrackbarPos("uniqRatio", "depth")
            swsize = cv2.getTrackbarPos("swsize", "depth")
            sprange = cv2.getTrackbarPos("sprange", "depth")
            mind = cv2.getTrackbarPos("mind", "depth")
            dispmd = cv2.getTrackbarPos("dispmd", "depth")
            stereo.setBlockSize(blockSize)
            stereo.setNumDisparities(num*16+16)
            stereo.setPreFilterCap(prefilp)
            stereo.setMinDisparity(mind)
            stereo.setUniquenessRatio(uniqRatio)
            stereo.setTextureThreshold(ttthread)
            stereo.setSpeckleWindowSize(swsize)
            stereo.setSpeckleRange(sprange)
            stereo.setDisp12MaxDiff(dispmd) 
            disparity = stereo.compute(iml_rectified_l, imr_rectified_r)
            filtered_disparity = cv2.medianBlur(disparity, 5)
            disp = cv2.normalize(filtered_disparity, filtered_disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
            # 将图片扩展至3d空间中，其z方向的值则为当前的距离
            threeD = cv2.reprojectImageTo3D(filtered_disparity, Q,True)
            threeD = threeD*16
            def callbackFunc(e, x, y, f, p):
                if e == cv2.EVENT_LBUTTONDOWN:        
                    print('点 (%d, %d) 的三维坐标 (%fmm, %fmm, %fmm)' % (x, y, threeD[y, x, 0], threeD[y, x, 1], threeD[y, x, 2]))
                    distance  = ( (threeD[y, x, 0] ** 2 + threeD[y, x, 1] ** 2 + threeD[y, x, 2] **2) ** 0.5) / 10
                    print('点 (%d, %d) 距离左摄像头的相对距离为 %0.3f cm, %0.3f cm' %(x, y,  distance,abs(threeD[y, x,2])/10.0) )
            cv2.setMouseCallback("depth", callbackFunc, None)
            cv2.imshow("left", iml_rectified_l)
            cv2.imshow("right", imr_rectified_r)
            cv2.imshow("depth", disp)
            cv2.imshow("color_depth",disp_color)
            key = cv2.waitKey(1)
            if key ==ord("s"):
                saved_file = {
                    "NumDisparities": num*16+16,
                    "BlockSize": blockSize,
                    "PreFilterCap": prefilp,
                    "UniquenessRatio": uniqRatio,
                    "TextureThreshold": ttthread,
                    "SpeckleWindowSize": swsize,
                    "SpeckleRange": sprange,
                    "MinDisparity": mind,
                    "Disp12MaxDiff": dispmd,
                }
                b = json.dumps(saved_file)
                f2 = open(saved_json_filepath, 'w')
                f2.write(b)
                f2.close()
                print("Stereo Match Param Saved!\n")
                print("NumDisparities: ",num*16+16)
                print("BlockSize: ",blockSize)
                print("PreFilterCap: ",prefilp)
                print("UniquenessRatio: ",uniqRatio)
                print("TextureThreshold: ",ttthread)
                print("SpeckleWindowSize: ",swsize)
                print("SpeckleRange: ",sprange)
                print("MinDisparity: ",mind)
                print("Disp12MaxDiff: ",dispmd)
            if key == ord("q"):
                print("NumDisparities: ",num*16+16)
                print("BlockSize: ",blockSize)
                print("PreFilterCap: ",prefilp)
                print("UniquenessRatio: ",uniqRatio)
                print("TextureThreshold: ",ttthread)
                print("SpeckleWindowSize: ",swsize)
                print("SpeckleRange: ",sprange)
                print("MinDisparity: ",mind)
                print("Disp12MaxDiff: ",dispmd)
                break
    cap.release()
    cv2.destroyAllWindows()
