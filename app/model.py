import mmdet
from mmcv import Config
from mmdet.apis import init_detector, inference_detector

def load_model():
    cfg = Config.fromfile('./configs/_base_/models/cascade_rcnn_r50_fpn.py.py')
    for i in cfg.model.roi_head.bbox_head:
        i['num_classes'] = 1

    checkpoint_path = 'detection_model/epoch_4.pth'
    detector_model = init_detector(cfg, checkpoint=checkpoint_path, device='gpu')
    return detector_model

def get_prediction(model, img):
    result = inference_detector(model, img)[0]
    return result