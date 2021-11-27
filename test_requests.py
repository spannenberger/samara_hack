import cv2
import os
import requests
import pandas as pd
import tqdm

URL = 'http://10.10.67.145:5010/api/test'

content_type = 'image/jpeg'
headers = {'content-type': content_type}


def get_recognize_princess(path):
    df = pd.DataFrame()
    df['id'] = os.listdir(path)
    df['class'] = None
    for idx, photo in enumerate(df['id']):
        image_array = cv2.imread(f'{path}/{photo}')
        # image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        _, img_encoded = cv2.imencode('.jpg', image_array)
        data = img_encoded.tostring()
        response = requests.post(URL, data=data, headers=headers)
        metadata = response.json()['image']
        for recog in metadata['bbox']:
            if recog['class_name'] == 'princess':
                df.loc[idx, 'class'] = 1
                continue
            else:
                print(photo)
                df.loc[idx, 'class'] = 0

        df.to_csv('labels_princess.csv', index=True)


def get_recognize_leotigers(path):
    df = pd.DataFrame()
    df['id'] = os.listdir(path)
    df['class'] = None
    for idx, photo in enumerate(df['id'][:10]):
        image_array = cv2.imread(f'{path}/{photo}')
        # image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        # здесь нужен try
        _, img_encoded = cv2.imencode('.jpg', image_array)
        data = img_encoded.tostring()
        response = requests.post(URL, data=data, headers=headers)
        metadata = response.json()['image']
        for recog in metadata['bbox']:
            if recog['class_name'] == 'tigers':
                df.loc[idx, 'class'] = 1
                continue
            elif recog['class_name'] == 'leopard':
                df.loc[idx, 'class'] = 2
                continue
            else:
                df.loc[idx, 'class'] = 3

        df.to_csv('labels_leotigers.csv', index=True)


if __name__=="__main__":
    get_recognize_princess('/home/egor-m/minprirodi/Принцесса_400/')
    get_recognize_leotigers(path)