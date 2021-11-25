from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import mmdet
from mmcv import Config
from mmdet.apis import init_detector, inference_detector

cfg = Config.fromfile('./configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py')
for i in cfg.model.roi_head.bbox_head:
    i['num_classes'] = 1

checkpoint_path = 'model/best_checkpoint.pth'

detector_model = init_detector(cfg, checkpoint=checkpoint_path, device='cpu')
print(detector_model)

app = Flask(__name__)


# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = inference_detector(detector_model, img)[0]
    print('detecting id done')

    all_bboxes = []
    for i, bbox in enumerate(result):
        if bbox[-1] > 0.25:
            all_bboxes.append({'bbox_id':i, 'bbox':{'x1':int(bbox[0]), 'y1':int(bbox[1]),\
                                                    'x2':int(bbox[2]), 'y2':int(bbox[3])}})

    response = {'message' : 'image received. size={}x{}'.format(img.shape[1], img.shape[0]),
                'image' : {'bbox':all_bboxes}
                }
    return response

# start flask app
app.run(host="0.0.0.0", port=5000)
