import os

os.getenv('./test/test')
from test.test.model_config import get_config
from detectron2.data.detection_utils import read_image
from point_rend import add_pointrend_config, point_model
import tensorflow as tf
import queue


ROOT_DIR = '/test/test/model/'
or_image = read_image('/test/test//SKIN/AK/ISIC_0024468.jpg', format="BGR")  # 对应的就是一张空白的图像，跟测试图像大小相同


class QueueObject():

    def __init__(self, queue, auto_get=False):
        self._queue = queue
        self.object = self._queue.get() if auto_get else None

    def __enter__(self):
        if self.object is None:
            self.object = self._queue.get()
        return self.object

    def __exit__(self, Type, value, traceback):
        if self.object is not None:
            self._queue.put(self.object)
            self.object = None

    def __del__(self):
        if self.object is not None:
            self._queue.put(self.object)
            self.object = None


def setup_cfg():
    cfg = get_config()
    add_pointrend_config(cfg)
    cfg.merge_from_file("/root/PointDetect/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = detection_MODEL_PATH
    cfg.freeze()
    return cfg


def get_detect_model():
    print("====================loading    detect    model==================")
    global detect_graph
    detect_graph = tf.get_default_graph()
    with detect_graph.as_default():
        config = setup_cfg()
        model_detect = point_model.LoadModel(config)
        model_detect(or_image)
    return model_detect


print("模型池启用,模型加载：")
# 实例化类
detec_model = queue.Queue()
for i in range(1):
    detec_model.put(get_detect_model())
detec_object = QueueObject(detec_model)

with detec_object as obj:
    #    print(detec_object)
    pass

print("模型加载完成。")