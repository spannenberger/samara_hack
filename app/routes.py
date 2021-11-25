from flask import request, render_template
import numpy as np
import cv2
from app import app
from .model import load_model, get_prediction

model = load_model
# route http posts to this method
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/api/test', methods=['POST'])
def test():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = get_prediction(model, img)
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
# app.run(host="0.0.0.0", port=5010)
