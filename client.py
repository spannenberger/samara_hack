# import cv2
# import os
# import requests
# import pandas as pd
# import tqdm

import datefinder

string_with_dates = '2016-05-20 16:50:15 M 3/73\n6 enasdASdaSdASdASDasDASd ee ee Es\n\x0c'

matches = datefinder.find_dates(string_with_dates)
for match in matches:
    date = match
    break
print(date)

# URL = 'http://10.10.67.145:5010/api/test'
#
# content_type = 'image/jpeg'
# headers = {'content-type': content_type}
# df = pd.DataFrame()
#
# for photo in tqdm.tqdm(os.listdir(f'/home/egor-m/minprirodi/Принцесса_400')):
#     try:
#         image_array = cv2.imread(f'/home/egor-m/minprirodi/Принцесса_400/{photo}')
#         image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
#
#         _, img_encoded = cv2.imencode('.jpg', image_array)
#         data = img_encoded.tostring()
#         response = requests.post(URL, data=data, headers=headers)
#         metadata = response.json()['image']
#         for bbox in metadata['bbox']:
#             x = bbox['bbox']['x1']
#             y = bbox['bbox']['y1']
#             w = bbox['bbox']['x2'] - bbox['bbox']['x1']
#             h = bbox['bbox']['y2'] - bbox['bbox']['y1']
#             # image_array = image_array[y:y+h, x:x+w]
#             cv2.imwrite(f'/home/egor-m/metric_learning_dataset/princess/{photo}', image_array[y:y+h, x:x+w])
#         # break
#     except:
#         print(metadata)


