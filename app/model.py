import mmdet
from mmcv import Config
from mmdet.apis import init_detector, inference_detector

def load_model():
    cfg = Config.fromfile('./configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py')
    print(cfg)
    for i in cfg.model.roi_head.bbox_head:
        i['num_classes'] = 1

    checkpoint_path = 'detection_model/best_checkpoint.pth'
    detector_model = init_detector(cfg, checkpoint=checkpoint_path, device='cuda')
    return detector_model

def get_prediction(model, img):
    test = inference_detector(model, img)
    result = test[0][0]
    return result