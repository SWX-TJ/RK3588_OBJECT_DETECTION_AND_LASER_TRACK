from rknnlite.api import RKNNLite
import cv2
import numpy as np
import platform
from copy import copy

OBJ_THRESH = 0.05
NMS_THRESH = 0.05
IMG_SIZE = (640, 640)
MESUREMENT_MIN_DIS = 10 #cm
MESUREMENT_MAX_DIS = 300 #cm
#这个地方可以替换为自定义训练的数据集的标签


#自定义检测使用的是哪个模型
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'
def get_host():
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                elif 'rk3562' in device_compatible_str:
                    host = 'RK3562'
                else:
                    host = 'RK3566_RK3568'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    # candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def softmax(x, axis=None):
    # 为了防止指数函数的数值溢出问题，减去每个元素所在轴的最大值
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    # 计算 softmax
    return e_x / e_x.sum(axis=axis, keepdims=True)
def dfl(position):
    # Distribution Focal Loss (DFL)
    # import torch
    # x = torch.tensor(position)
    n,c,h,w = position.shape
    p_num = 4
    mc = c//p_num
    y = position.reshape(n,p_num,mc,h,w)
    y =softmax(y,2) #y.softmax(2)
    #print("range(mc)", range(mc))
    acc_metrix  = np.arange(mc).reshape(1, 1, mc, 1, 1)#= range(mc).reshape(1,1,mc,1,1)#torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[0]//grid_h, IMG_SIZE[1]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)
    return xyxy

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores


# def draw(image, boxes, scores, classes):
#     for box, score, cl in zip(boxes, scores, classes):
#         top, left, right, bottom = [int(_b) for _b in box]
#         #print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
#         cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
#         cv2.circle(image,((top+right)//2,(left+bottom)//2),2,(0, 255, 0),2)
#         cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
#                     (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        
# def draw_depth(image, boxes, scores, classes,depthimage):
#     for box, score, cl in zip(boxes, scores, classes):
#         top, left, right, bottom = [int(_b) for _b in box]
#         #print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
#         center_x = (top+right)//2
#         center_y= (left+bottom)//2
#         #print("center_x",center_x,center_y,depthimage.shape)
#         dis = ((depthimage[center_y, center_x, 0] ** 2 + depthimage[center_y, center_x, 1] ** 2 + depthimage[center_y, center_x, 2] **2) ** 0.5) / 10
#         if cl!=0:
#             cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
#             if dis<=200 and dis>=18:  #
#                 cv2.circle(image,((top+right)//2,(left+bottom)//2),2,(0, 255, 0),2)
#                 cv2.putText(image, '{0} {1:.2f} cm'.format(CLASSES[cl], dis),
#                         (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                


def is_point_in_box(point, top,left,right,bottom):
    (x, y) = point
    if x>=top and x<=right and y>=left and y<=bottom:
        return True
    return False#

def justdrawdetectionresult(image, boxes, scores, classes,depthimage,ObjClassNameList):
    for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = [int(_b) for _b in box]
            #print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
            center_x = (top+right)//2
            center_y= (left+bottom)//2
            #print("center_x",center_x,center_y,depthimage.shape)
            dis = ((depthimage[center_y, center_x, 0] ** 2 + depthimage[center_y, center_x, 1] ** 2 + depthimage[center_y, center_x, 2] **2) ** 0.5) / 10
            if cl!=0:
                print("objclass",ObjClassNameList[cl])
                print("score",score,dis)
                cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
                if dis<=MESUREMENT_MAX_DIS and dis>=MESUREMENT_MIN_DIS:  #
                    cv2.circle(image,((top+right)//2,(left+bottom)//2),2,(0, 0, 255),2)
                    cv2.putText(image, '{0}'.format(ObjClassNameList[cl]),
                            (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    # return image

def selecttrackedobjinfo(image, boxes, scores, classes,depthimage,clicked_object_cord,ObjClassNameList):
    tracked_object_cls = None
    tracked_world_pos = (0,0,0)
    tracked_bbox = (-1,-1,-1,-1)
    for box,socre,cl in zip(boxes,scores,classes):
        if ObjClassNameList[cl]!="person":  #refused person detection!
            top, left, right, bottom = [int(_b) for _b in box]
            center_x = (top+right)//2
            center_y= (left+bottom)//2
            dis = ((depthimage[center_y, center_x, 0] ** 2 + depthimage[center_y, center_x, 1] ** 2 + depthimage[center_y, center_x, 2] **2) ** 0.5) / 10
            if is_point_in_box(clicked_object_cord,top, left, right, bottom):
                cv2.rectangle(image, (top, left), (right, bottom), (0,255,0), 2)
                if dis<=MESUREMENT_MAX_DIS and dis>=MESUREMENT_MIN_DIS:  #有效距离
                    cv2.circle(image,((top+right)//2,(left+bottom)//2),2,(0, 0, 255),2)
                    cv2.putText(image, '{0}'.format(ObjClassNameList[cl]),
                            (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    tracked_object_cls = ObjClassNameList[cl]
                    tracked_world_pos = ((float)(depthimage[center_y, center_x, 0])/10.0,(float)(depthimage[center_y, center_x, 1])/10.0,(float)(depthimage[center_y, center_x, 2])/10.0)
                    tracked_bbox = (top,left,right-top,bottom-left)
                    break
    # print("tracked_object_cls-->",tracked_object_cls)
    # print("tracked_bbox-->",tracked_bbox)
    # print("tracked_world_pos-->",tracked_world_pos)
    return image,tracked_bbox,tracked_world_pos,tracked_object_cls


# def mainprocess_laser(image, boxes, scores, classes,depthimage,clicked_object_cord):
#     tracked_object_cls = None
#     tracked_pos = (0,0,0)
#     if clicked_object_cord[0]!=-100:  #user clicked object
#         for box,socre,cl in zip(boxes,scores,classes):
#             if cl!=0:  #refused person detection!
#                 top, left, right, bottom = [int(_b) for _b in box]
#                 center_x = (top+right)//2
#                 center_y= (left+bottom)//2
#                 dis = ((depthimage[center_y, center_x, 0] ** 2 + depthimage[center_y, center_x, 1] ** 2 + depthimage[center_y, center_x, 2] **2) ** 0.5) / 10
#                 if is_point_in_box(clicked_object_cord,top, left, right, bottom):
#                     cv2.rectangle(image, (top, left), (right, bottom), (0,255,0), 2)
#                     if dis<=MESUREMENT_MAX_DIS and dis>=MESUREMENT_MIN_DIS:  #
#                         cv2.circle(image,((top+right)//2,(left+bottom)//2),2,(0, 0, 255),2)
#                         cv2.putText(image, '{0}'.format(CLASSES[cl]),
#                                 (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
#                         tracked_object_cls = CLASSES[cl]
#                         tracked_pos = ((float)(depthimage[center_y, center_x, 0])/10.0,(float)(depthimage[center_y, center_x, 1])/10.0,(float)(depthimage[center_y, center_x, 2])/10.0)
#                 else:
#                     cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
#                     if dis<=MESUREMENT_MAX_DIS and dis>=MESUREMENT_MIN_DIS:  #
#                         cv2.circle(image,((top+right)//2,(left+bottom)//2),2,(0, 0, 255),2)
#                         cv2.putText(image, '{0}'.format(CLASSES[cl]),
#                                 (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
#             else: #person detection
#                 continue
#         return image,tracked_object_cls,tracked_pos
#     else:
#         for box, score, cl in zip(boxes, scores, classes):
#             top, left, right, bottom = [int(_b) for _b in box]
#             #print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
#             center_x = (top+right)//2
#             center_y= (left+bottom)//2
#             #print("center_x",center_x,center_y,depthimage.shape)
#             dis = ((depthimage[center_y, center_x, 0] ** 2 + depthimage[center_y, center_x, 1] ** 2 + depthimage[center_y, center_x, 2] **2) ** 0.5) / 10
#             # if cl!=0:
#             cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
#             if dis<=MESUREMENT_MAX_DIS and dis>=MESUREMENT_MIN_DIS:  #
#                 cv2.circle(image,((top+right)//2,(left+bottom)//2),2,(0, 0, 255),2)
#                 cv2.putText(image, '{0}'.format(CLASSES[cl]),
#                         (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
#         return image,tracked_object_cls,tracked_pos




def letter_box(im, new_shape, color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_CUBIC)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def get_real_box(src_shape, box, dw, dh, ratio):
    bbox = copy(box)
    # unletter_box result
    bbox[:,0] -= dw
    bbox[:,0] /= ratio
    bbox[:,0] = np.clip(bbox[:,0], 0, src_shape[1])

    bbox[:,1] -= dh
    bbox[:,1] /= ratio
    bbox[:,1] = np.clip(bbox[:,1], 0, src_shape[0])

    bbox[:,2] -= dw
    bbox[:,2] /= ratio
    bbox[:,2] = np.clip(bbox[:,2], 0, src_shape[1])

    bbox[:,3] -= dh
    bbox[:,3] /= ratio
    bbox[:,3] = np.clip(bbox[:,3], 0, src_shape[0])
    return bbox

# def YoloObjectDetection(rknn_lite,ori_img):
#     src_shape = ori_img.shape[:2]
#     img, ratio, (dw, dh) = letter_box(ori_img, IMG_SIZE)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_tensor = np.expand_dims(img, 0)
#     outputs = rknn_lite.inference(inputs=[img_tensor], data_format=['nhwc'])
#     boxes, classes, scores = post_process(outputs)
#     img_p = ori_img.copy()
#     if boxes is not None:
#         draw(img_p, get_real_box(src_shape, boxes, dw, dh, ratio), scores, classes)


def YoloObjectDetectionandMeasureDistance(rknn_lite,stereoMatcher,ori_image_left,ori_image_right):
    depthImage,iml_rectified_l,disparity_color = stereoMatcher.CacuDepth(ori_image_left,ori_image_right,True)
    src_shape = iml_rectified_l.shape[:2]
    img, ratio, (dw, dh) = letter_box(iml_rectified_l, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = np.expand_dims(img, 0)
    outputs = rknn_lite.inference(inputs=[img_tensor], data_format=['nhwc'])
    boxes, classes, scores = post_process(outputs)
    img_p = iml_rectified_l.copy()
    if boxes is not None:
        real_boxes= get_real_box(src_shape, boxes, dw, dh, ratio)
        return (real_boxes, classes, scores,depthImage,img_p)
    return (None,classes,scores,depthImage,img_p)


# def YoloObjectDetectionandMeasureDistance_mutiimage(rknn_lite,stereoMatcher,ori_image_left,ori_image_right):
#     depthImage,iml_rectified_l,_ = stereoMatcher.CacuDepth(ori_image_left,ori_image_right,True)
#     img = cv2.cvtColor(iml_rectified_l, cv2.COLOR_BGR2RGB)
#     src_shape = img.shape[:2]
#     mulpicplus = 2


    # img, ratio, (dw, dh) = letter_box(iml_rectified_l, IMG_SIZE)
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # img_tensor = np.expand_dims(img, 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # #图像分块
    # mulpicplus = 2
    # img_tensor = np.expand_dims(iml_rectified_l, 0)
    # print("img_tensor",img_tensor.shape)
    # xsz = img_tensor.shape[2]
    # ysz = img_tensor.shape[1]
    # x_smalloccur = int(xsz / mulpicplus * 1.2)
    # y_smalloccur = int(ysz / mulpicplus * 1.2)
    # for i in range(mulpicplus):
    #     x_startpoint = int(i * (xsz / mulpicplus))
    #     for j in range(mulpicplus):
    #         y_startpoint = int(j * (ysz / mulpicplus))
    #         x_real = min(x_startpoint + x_smalloccur, xsz)
    #         y_real = min(y_startpoint + y_smalloccur, ysz)
    #         if (x_real - x_startpoint) % 64 != 0:
    #             x_real = x_real - (x_real-x_startpoint) % 64
    #         if (y_real - y_startpoint) % 64 != 0:
    #             y_real = y_real - (y_real - y_startpoint) % 64
    #         dicsrc = img_tensor[:, y_startpoint:y_real, x_startpoint:x_real,:]
    #         print("dicsrc",dicsrc.shape)
    #         disc_output = rknn_lite.inference(inputs=[dicsrc], data_format=['nhwc'])
    #         disc_boxes, disc_classes, disc_scores = post_process(disc_output)
    #         if disc_boxes is not None:
    #            disc_boxes[:,0] =  disc_boxes[:,0]+y_startpoint  #top
    #            disc_boxes[:,1] =  disc_boxes[:,1]+x_startpoint  #left
    #            disc_boxes[:,2] =  disc_boxes[:,2]+y_startpoint  #bottom
    #            disc_boxes[:,3] =  disc_boxes[:,3]+x_startpoint  #right
    #         if i==0 and j == 0:
    #             pred_disc_boxes = disc_boxes
    #             pred_disc_classes = disc_classes
    #             pred_disc_scores = disc_scores
    #         else:
    #             pred_disc_boxes = np.concatenate([pred_disc_boxes, disc_boxes], dim=0)
    #             pred_disc_classes = np.concatenate([pred_disc_classes, disc_classes], dim=0)
    #             pred_disc_scores = np.concatenate([pred_disc_scores, disc_scores], dim=0)
    # img_p = iml_rectified_l.copy()
    # if pred_disc_boxes is not None:
    #     print("boxes",len(pred_disc_boxes),pred_disc_boxes.shape)
    #     real_boxes= get_real_box(src_shape, pred_disc_boxes, dw, dh, ratio)
    #     return (real_boxes, pred_disc_classes, pred_disc_scores,depthImage,img_p)
    # return (None,pred_disc_classes,pred_disc_scores,depthImage,img_p)