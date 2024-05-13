import cv2
import numpy as np
import json

#双目匹配类
class StereoMatcher(object):
    def __init__(self,CameraParams,stereobm_param_filepath='./stereobm.json',mode='BM'):
        self.mode = mode
        self.CameraParams = CameraParams
        self.img_channels = 3
        #1.先进行立体匹配和矫正
        self.map1x, self.map1y, self.map2x, self.map2y, self.Q, self.roi1, self.roi2 = self.PreStereoRectify(CameraParams)
        #2.根据mode选择使用SGBM还是BM
        if mode=='BM':
            self.StereoBMMatcher = self.CreateStereoBMMatcher(self.roi1, self.roi2,stereobm_param_filepath)
            #如果是bm,可以选择创建中值滤波器补充空洞
        elif mode=='SGBM':
            self.LeftStereoSGBMMatcher,self.RightStereoSGBMMatcher = self.CreateStereoSGBMMatcher(stereobm_param_filepath)
            #如果是sgbm,可以选择创建WLS滤波器补充空洞
            wsl_sigma = 1.5
            wsl_lmbda = 8000.0
            self.WLS_filter = cv2.ximgproc.createDisparityWLSFilter(self.LeftStereoSGBMMatcher)
            self.WLS_filter.setLambda(wsl_lmbda)
            self.WLS_filter.setSigmaColor(wsl_sigma)
    def preprocess(self,img1, img2):
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
        equalized_img1 = adjust_contrast(denoised_img1,1.2)#cv2.equalizeHist(denoised_img1)
        equalized_img2 = adjust_contrast(denoised_img2,1.2) #cv2.equalizeHist(denoised_img2)
        return equalized_img1, equalized_img2
    def CacuDepth(self,left_image,right_image,isshowdepthcolor=True):
        #1.去畸变
        left_image =  self.undistortion(left_image, self.CameraParams.cam_matrix_left, self.CameraParams.distortion_l)
        right_image =  self.undistortion(right_image, self.CameraParams.cam_matrix_right, self.CameraParams.distortion_r)
        #left_image_p,right_image_p =self.preprocess(left_image,right_image)
        #2.立体校正
        iml_rectified_l, imr_rectified_r = self.rectifyImage(left_image, right_image, self.map1x, self.map1y, self.map2x, self.map2y)
        #3.预处理
        #3.灰度化
        if iml_rectified_l.ndim==3:
            iml_rectified_l_gray,imr_rectified_r_gray = self.preprocess(iml_rectified_l,imr_rectified_r)#cv2.cvtColor(iml_rectified_l, cv2.COLOR_BGR2GRAY)
        else:
            iml_rectified_l_gray = iml_rectified_l.copy()
            imr_rectified_r_gray = imr_rectified_r.copy()
        if self.mode=='BM':
            disparity = self.StereoBMMatcher.compute(iml_rectified_l_gray, imr_rectified_r_gray)
            #filtered_disparity= disparity
            filtered_disparity = cv2.medianBlur(disparity, 5)
            #DepthImage= None
            DepthImage = cv2.reprojectImageTo3D(filtered_disparity,self.Q,True)
            DepthImage= DepthImage*16
            #可选（如果不想显示深度图，就可以设置isshowdepthcolor=False）
            if isshowdepthcolor:
                disparity_norm = cv2.normalize(filtered_disparity, filtered_disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                disparity_color = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_JET)
                return DepthImage,iml_rectified_l,disparity_color
            else:
                return DepthImage,iml_rectified_l,None
        elif self.mode=='SGBM':
            disparity_left = self.LeftStereoSGBMMatcher.compute(iml_rectified_l_gray, imr_rectified_r_gray)
            disparity_right = self.RightStereoSGBMMatcher.compute(imr_rectified_r_gray, iml_rectified_l_gray)
            trueDisp_left = disparity_left.astype(np.float32) / 16.
            trueDisp_right = disparity_right.astype(np.float32) / 16.
            filtered_disparity= self.WLS_filter.filter(trueDisp_left,iml_rectified_l_gray,None,trueDisp_right)
            DepthImage = cv2.reprojectImageTo3D(filtered_disparity,self.Q,True)
             #可选（如果不想显示深度图，就可以设置isshowdepthcolor=False）
            if isshowdepthcolor:
                disparity_norm = cv2.normalize(filtered_disparity, filtered_disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                disparity_color = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_JET)
                return DepthImage,iml_rectified_l,disparity_color
            else:
                return DepthImage,iml_rectified_l,None
        else:
            return None,left_image,None
    # 消除畸变
    def undistortion(self,image, camera_matrix, dist_coeff):
        undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)
        return undistortion_image
    # 畸变校正和立体校正
    def rectifyImage(self,image1, image2, map1x, map1y, map2x, map2y):
        rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
        rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
        return rectifyed_img1, rectifyed_img2
    def PreStereoRectify(self,CameraParams):
        # 读取内参和外参
        left_K = CameraParams.cam_matrix_left
        right_K = CameraParams.cam_matrix_right
        left_distortion = CameraParams.distortion_l
        right_distortion = CameraParams.distortion_r
        R = CameraParams.R
        T = CameraParams.T
        width = CameraParams.image_width
        height = CameraParams.image_height
        # 计算校正变换
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, 
                                                        (width, height), R, T, alpha=0)
        map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
        return map1x, map1y, map2x, map2y, Q, roi1, roi2
    def CreateStereoBMMatcher(self,validPixROI1,validPixROI2,stereobm_param_filepath):
        f = open(stereobm_param_filepath, 'r')
        content = f.read()
        BMParams = json.loads(content)
        f.close()
        StereoBMMatcher = cv2.StereoBM_create(numDisparities=BMParams['NumDisparities'], blockSize=BMParams['BlockSize'])
        StereoBMMatcher.setROI1(validPixROI1)
        StereoBMMatcher.setROI2(validPixROI2)
        StereoBMMatcher.setPreFilterCap(BMParams['PreFilterCap'])
        StereoBMMatcher.setMinDisparity(BMParams['MinDisparity'])
        StereoBMMatcher.setUniquenessRatio(BMParams['UniquenessRatio'])
        StereoBMMatcher.setTextureThreshold(BMParams['TextureThreshold'])
        StereoBMMatcher.setSpeckleWindowSize(BMParams['SpeckleWindowSize'])
        StereoBMMatcher.setSpeckleRange(BMParams['SpeckleRange'])
        StereoBMMatcher.setDisp12MaxDiff(BMParams['Disp12MaxDiff'])
        return StereoBMMatcher
    def CreateStereoSGBMMatcher(self,stereobm_param_filepath):
        f = open(stereobm_param_filepath, 'r')
        content = f.read()
        SGBMParams = json.loads(content)
        f.close()
        paraml = {'minDisparity': SGBMParams['minDisparity'],
                        'numDisparities':SGBMParams['numDisparities'],
                        'blockSize':SGBMParams['blockSize'],
                        'P1': SGBMParams['P1'],
                        'P2': SGBMParams['P2'],
                        'disp12MaxDiff': SGBMParams['disp12MaxDiff'],
                        'preFilterCap': SGBMParams['preFilterCap'],
                        'uniquenessRatio':  SGBMParams['uniquenessRatio'],
                        'speckleWindowSize':SGBMParams['speckleWindowSize'],
                        'speckleRange': SGBMParams['speckleRange'],
                        'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
                        }
        # 构建SGBM对象
        LeftStereoSGBMMatcher = cv2.StereoSGBM_create(**paraml)
        paramr = paraml.copy()
        paramr['minDisparity'] = -paraml['numDisparities']
        RightStereoSGBMMatcher = cv2.StereoSGBM_create(**paramr)
        return LeftStereoSGBMMatcher,RightStereoSGBMMatcher

