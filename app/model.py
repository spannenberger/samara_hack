import mmdet
from mmcv import Config
from mmdet.apis import init_detector, inference_detector
from transformers import ViTFeatureExtractor, ViTModel
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from PIL import Image
import torch

def load_detection_model():
    """Функция для загрузки обученной модели детекции"""

    cfg = Config.fromfile('./detection_model/config.py')

    checkpoint_path = 'detection_model/latest.pth'
    detector_model = init_detector(cfg, checkpoint=checkpoint_path, device='cuda')
    return detector_model


def load_metric_model():
    """Загрузки обученной модели metric learning 

    Модель metric learning используется для определения принцессы
    """

    extractor = 'google/vit-base-patch16-384' # фичи экстрактор
    model = './trained_metric_transformer/' # путь до обученной модели
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # пока пусть так будет, нужно сделать не так явно

    df = pd.read_csv("./base_embs.csv") # считываем csv со средними эмбеддингами для каждого класса
    base = np.array(df['0'])

    feature_extractor = ViTFeatureExtractor.from_pretrained(extractor)
    metric_model = ViTModel.from_pretrained(model)
    metric_model.to(device)

    return metric_model, feature_extractor, device, base


def get_metric_prediction(
    model, 
    feature_extractor, 
    device, 
    base, 
    img):
    """ Функция инференса metric learning

    Прогоняем фотографию через модель metric learning для получения его эмбеддинга
    Смотрим косинусное расстояние до эталонного эмбеддинга запредикченного класса
    (Далле по найденному threshold мы будем отсекать фотографии где нет принцессы)
    """
    
    img = feature_extractor(img,return_tensors="pt")
    img.to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(**img).pooler_output
        prediction = prediction[0].cpu().detach().numpy()

    dist = cosine(base, prediction) # считаем косинусное расстояние

    return dist


def get_detection_prediction(model, img):
    """"Функция для инференса детектора"""

    result = inference_detector(model, img)

    return result

