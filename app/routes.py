from .model import load_detection_model, get_detection_prediction, load_metric_model, get_metric_prediction
from flask import request, render_template
import numpy as np
import cv2
from app import app
import torch
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def get_image_from_tg_bot(request_from_bot):
    """Функция для обрабтки фотографии с тг запроса
    Args:
        request_from_bot: bytearray - байтовое представление изображения
    Return:
        img: np.array - numpy массив-представление изображения
    """

    nparr = np.fromstring(request_from_bot.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


detection_model = load_detection_model() # выгрузка детектора
metric_model, feature_extractor, device, base = load_metric_model() # выгрузка metric learning модели

@app.route('/')
@app.route('/index')
def index():
    """Рендеринг html страницы

    Показательная страница, показывающая возможность простого расширения и внедрения сервиса
    """

    return render_template('index.html')

@app.route('/api/test', methods=['POST'])
def test():

    img = get_image_from_tg_bot(request)

    detection_result = get_detection_prediction(detection_model, img) # получение результатов детекции
    print('detecting is done')

    all_bboxes = []
    for i, recognition in enumerate(detection_result):

        if recognition.shape[0] == 0: # проверка результата детектора, если по [0] ничего, мы его пропускаем
            continue

        for bbox in recognition:
            print(bbox[-1])
            if bbox[-1] > 0.8: # подобранный threshold для точности детекции
                all_bboxes.append({'bbox_id':i, 'bbox':{'x1':int(bbox[0]), 'y1':int(bbox[1]),\
                                                        'x2':int(bbox[2]), 'y2':int(bbox[3])},\
                                   'threshold':int(bbox[-1]*100)})
        for bbox in all_bboxes:
            topLeftCorner = (bbox['bbox']['x1'], bbox['bbox']['y1'])
            botRightCorner = (bbox['bbox']['x2'], bbox['bbox']['y2'])

            # обрезаем фотографию по bbox
            cutted_img = cv2.rectangle(img,\
                topLeftCorner,\
                botRightCorner,\
                (255, 0, 0), 1)

            # получаем предикт с metric learning модели
            metric_result = get_metric_prediction(metric_model, feature_extractor, device, base, cutted_img)

            classes_dict = {0: 'leopard', 1: 'princess', 2: 'tigers', 3: 'other animal'}

            bbox.update({'class_name': classes_dict[metric_result]})
            print(metric_result)

    response = {'message' : 'image received. size={}x{}'.format(img.shape[1], img.shape[0]),
                'image' : {'bbox':all_bboxes}
                }

    return response
