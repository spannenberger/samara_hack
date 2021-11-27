import io
import telegram
from PIL import Image
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters
from telegram.ext import Updater
from credentials import bot_token
import cv2
import numpy as np
import requests
import os 
import glob
from telegram import InputMediaPhoto

TOKEN = bot_token
URL = 'http://10.10.67.145:5010/api/test'

content_type = 'image/jpeg'
headers = {'content-type': content_type}

updater = Updater(token=TOKEN)
dispatcher = updater.dispatcher

print('initializing completed')


def start(bot, update):
    """Стартовая команда для бота"""

    text = 'Привет, я бот распознающий животных в заповедниках, пришли мне фото и я отрисую всех животных, ' \
           'а так же выведу виды этих животных \n'
    bot.send_message(chat_id=update.message.chat_id, text=text)


def cut_video(bot, update):
    text = 'Также я могу сделать распознание по видеофрагменту\n'
    bot.send_message(chat_id=update.message.chat_id, text=text)


def process_image(image_bytes):
    """Перевод изображения из байтов в необходимый для вывода формат"""

    image = np.asarray(bytearray(image_bytes), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def transform_pil_image_to_bytes(image):
    """Перевод изображения в байты"""

    image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)

    return buffer


def get_image_bytes(bot, update):
    if not update.message.photo:
        file_id = update.message.document['file_id']
    else:
        file_id = update.message.photo[-1]['file_id']
    file = bot.getFile(file_id)
    image = file.download_as_bytearray()
    return image, file_id


def get_video(bot, update):
    # import pdb;pdb.set_trace()
    if not update.message.video:
        file_id = update.message.document['file_id']
    else:
        file_id = update.message.video['file_id']
    file = bot.getFile(file_id)
    # 
    file_path = file.file_path
    return file_path


def draw_contours(image_array, metadata):
    """Отрисовка контуров по bbox и подсчет кол-ва найденных особей"""

    counter_dict = {'leopard': 0, 'tigers': 0, 'other animal': 0, 'is_princess': False}
    for bbox in metadata['bbox']:
        import pdb;pdb.set_trace()
        class_name = bbox['class_name']
        if class_name == 'princess':
            counter_dict['is_princess'] = True
            counter_dict['tigers'] += 1
        else:
            counter_dict[class_name] += 1
        threshold = bbox['threshold']

        topLeftCorner = (bbox['bbox']['x1'], bbox['bbox']['y1'])
        botRightCorner = (bbox['bbox']['x2'], bbox['bbox']['y2'])

        cv2.rectangle(image_array,\
                         topLeftCorner,\
                         botRightCorner,\
                         (255, 0, 0), 1)

        cv2.putText(image_array, f'{class_name} - {threshold}',
                        topLeftCorner,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                        2)

    return counter_dict

def bot_video_preprocessing(bot, update):
    text = 'Начинаю обработку видеофрагмента\n'
    bot.send_message(chat_id=update.message.chat_id, text=text)

    file_path = get_video(bot, update)

    cap = cv2.VideoCapture(file_path)
    photo_count = 0
    os.makedirs('/workspace/bot/photosfrom_video', exist_ok=True)
    text = 'Считал ваше видео, начинаю поиск полезных кадров\n'
    bot.send_message(chat_id=update.message.chat_id, text=text)
    if not cap.isOpened():
        print('Cannot to open video file')
    count = 0
    while cap.isOpened():

        fl, img_frame = cap.read()
        if img_frame is None:
            break

        photo_count += 1
        image_array = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)

        text = 'О.... Кажется, нашел один.... Еще секундочку\n'
        bot.send_message(chat_id=update.message.chat_id, text=text)
        _, img_encoded = cv2.imencode('.jpg', image_array)
        data = img_encoded.tostring()

        response = requests.post(URL, data=data, headers=headers)
        metadata = response.json()['image']
        leos_count, tigers_count, other_animal_count, is_princess = draw_contours(image_array, metadata).values()

        cv2.imwrite(f'/workspace/bot/photosfrom_video/photo {photo_count}.jpg', cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

        count += 120 # 1 секунда = 30; 2 секунды = 60 и тд

        cap.set(cv2.CAP_PROP_POS_FRAMES, count)

    text = 'Обработку закончил, все в норме! Сейчас отправлю альбом фотографий)\n'
    bot.send_message(chat_id=update.message.chat_id, text=text)

    media_group = []
    text = 'some caption for album'
    for img in glob.glob("/workspace/bot/photosfrom_video/*.jpg"):
        media_group.append(InputMediaPhoto(open(img, 'rb')))

    bot.send_media_group(chat_id=update.message.chat_id, media = media_group)


def bot_image_processing(bot, update):
    """Функция обработка запроса с изображением
    
    Получаем и обрабатываем запрос с бэка 
    Формируем сообщение в бота
    """

    image_bytes, file_id = get_image_bytes(bot, update)

    image_array = process_image(image_bytes)
    _, img_encoded = cv2.imencode('.jpg', image_array)

    data = img_encoded.tostring()
    response = requests.post(URL, data=data, headers=headers)

    metadata = response.json()['image']
    # detected_date = response.json()['date']
    # import pdb;pdb.set_trace()
    leos_count, tigers_count, other_animal_count, is_princess = draw_contours(image_array, metadata).values()
    image = transform_pil_image_to_bytes(image_array)

    print('vse ok')
    if is_princess:
        text = f"На фото обнаружили \nЛеопард🐆 - {leos_count} \nТигр🐯 - {tigers_count} \nДругие животные - {other_animal_count}🎁 \nТакже на фото была обнаружена принцесска 👸 👑."
    else:
        text = f"На фото обнаружили \nЛеопард🐆 - {leos_count} \nТигр🐯 - {tigers_count} \nДругие животные - {other_animal_count}🎁 \nПринцессы обнаружено не было..."
    bot.send_photo(chat_id=update.message.chat_id, photo=image)
    bot.send_message(chat_id=update.message.chat_id, text=text,
                    reply_to_message_id=update.message.message_id,
                    parse_mode=telegram.ParseMode.HTML)





start_handler = CommandHandler(['start', 'help'], start)
cut_video_handler = CommandHandler(['cut_video'], cut_video)

process_image_handler = MessageHandler(Filters.photo | Filters.document, bot_image_processing)
process_video_handler =  MessageHandler(Filters.video, bot_video_preprocessing)

dispatcher.add_handler(start_handler)
dispatcher.add_handler(cut_video_handler)
dispatcher.add_handler(process_image_handler)
dispatcher.add_handler(process_video_handler)
updater.start_polling()
