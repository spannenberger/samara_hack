from flask import request, render_template
import numpy as np
import cv2
from app import app
from .model import load_model, get_prediction
import pytesseract
from datetime import datetime
import datefinder
import re

import torch
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

custom_tesseract_config = '--oem 3 --psm 6 outputbase digits'


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def get_time_from_pytesseract(image_array):
    gray = get_grayscale(image_array)
    # gray = remove_noise(gray)
    # gray = thresholding(gray)

    gray_bot = gray[gray.shape[0] - 60:gray.shape[0], 0:gray.shape[1]]
    gray_top = gray[0:30, 0:gray.shape[1]]


    times_string_bot_image = pytesseract.image_to_string(gray_bot, config=custom_tesseract_config)
    print(times_string_bot_image)
    match_time = re.search(r'\d\d:\d\d:\d\d', times_string_bot_image)
    if not match_time:
        match_time = re.search(r'\d\d: \d\d: \d\d', times_string_bot_image)
    match_date = re.search(r'\d\d/\d\d/\d\d', times_string_bot_image)
    if not match_date:
        match_date = re.search(r'\d\d-\d\d-\d\d', times_string_bot_image)

    if (not match_time) or (not match_date):
        pass
    else:
        return match_date.group() + " " + match_time.group()

    times_string_top_image = pytesseract.image_to_string(gray_top, config=custom_tesseract_config)
    print(times_string_top_image)
    match_time = re.search(r'\d\d:\d\d:\d\d', times_string_top_image)
    if not match_time:
        match_time = re.search(r'\d\d: \d\d: \d\d', times_string_top_image)
    match_date = re.search(r'\d\d/\d\d/\d\d', times_string_top_image)
    if not match_date:
        match_date = re.search(r'\d\d-\d\d-\d\d', times_string_top_image)

    return match_date.group() + " " + match_time.group()


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
