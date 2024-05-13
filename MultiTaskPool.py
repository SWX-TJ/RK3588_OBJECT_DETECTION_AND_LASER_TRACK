from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor, as_completed
from dist.stereoMatcherCls import StereoMatcher#双目匹配类 

def initRKNN(rknnModel="rknn_model/yolov8.rknn", id=0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    if id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif id == -1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    print(rknnModel, "\t\tdone")
    return rknn_lite


def initRKNNs(rknnModel=".rknn_model/yolov8.rknn", TPEs=1):
    rknn_list = []
    for i in range(TPEs):
        rknn_list.append(initRKNN(rknnModel, i % 3))
    return rknn_list

def initSteroMatch(cameraParams,stereobmParams,TPEs=1):
    stereomatchs = []
    for i in range(TPEs):
        stereomatchs.append(StereoMatcher(cameraParams,stereobmParams,'BM'))
    return stereomatchs

class TaskPoolExecutor():
    def __init__(self, rknnModel, cameraParams,stereobmParams,TPEs, func):
        self.TPEs = TPEs
        self.queue = Queue()
        self.rknnPool = initRKNNs(rknnModel, TPEs)
        self.stereoPool = initSteroMatch(cameraParams,stereobmParams,TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0

    def put(self, left_img,right_img):
        self.queue.put(self.pool.submit(
            self.func, self.rknnPool[self.num % self.TPEs], self.stereoPool[self.num % self.TPEs],left_img,right_img))
        self.num += 1
 
    def get(self):
        if self.queue.empty():
            return None, False
        temp = []
        temp.append(self.queue.get())
        for frame in as_completed(temp):
            return frame.result(), True
 
    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()
