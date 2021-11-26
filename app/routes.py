from flask import request, render_template
import numpy as np
import cv2
from app import app
from .model import load_model, get_prediction
import pytesseract
from datetime import datetime
import datefinder

custom_tesseract_config = '--oem 3 --psm 6 outputbase digits'


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_time_from_pytesseract(image_array):
    gray = get_grayscale(image_array)
    gray_bot_right = gray[gray.shape[0] - 100:gray.shape[0], gray.shape[1] - 650:gray.shape[1]]
    gray_top_left = gray[0:30, 0:700]

    print('pytesseract with gray_bot_right')
    times_string = pytesseract.image_to_string(gray_bot_right, config=custom_tesseract_config)
    print('pytesseract with gray_top_left')
    print(pytesseract.image_to_string(gray_top_left, config=custom_tesseract_config))
    times_string += pytesseract.image_to_string(gray_top_left, config=custom_tesseract_config)

    matches = datefinder.find_dates(times_string)
    for match in matches:
        detected_date = match
        break
    return detected_date


def get_image_from_tg_bot(request_from_bot):
    nparr = np.fromstring(request_from_bot.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


model = load_model()
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/api/test', methods=['POST'])
def test():

    img = get_image_from_tg_bot(request)
    detected_date = get_time_from_pytesseract(img)

    result = get_prediction(model, img)
    print('detecting is done')

    all_bboxes = []
    for i, recognition in enumerate(result):
        if recognition.shape[0] == 0:
            continue
        for bbox in recognition:
            print(bbox[-1])
            if bbox[-1] > 0.92:
                all_bboxes.append({'bbox_id':i, 'bbox':{'x1':int(bbox[0]), 'y1':int(bbox[1]),\
                                                        'x2':int(bbox[2]), 'y2':int(bbox[3])},\
                                   'threshold':int(bbox[-1]*100)})
    
    response = {'message' : 'image received. size={}x{}'.format(img.shape[1], img.shape[0]),
                'image' : {'bbox':all_bboxes},
                'date':detected_date
                }
    return response
