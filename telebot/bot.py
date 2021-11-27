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


TOKEN = bot_token
URL = 'http://10.10.67.145:5010/api/test'

content_type = 'image/jpeg'
headers = {'content-type': content_type}

updater = Updater(token=TOKEN)
dispatcher = updater.dispatcher

print('initializing completed')


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


def start(bot, update):
    """Стартовая команда для бота"""

    text = 'Привет, я бот распознающий животных в заповедниках, пришли мне фото и я отрисую всех животных, ' \
           'а так же выведу виды этих животных \n'
    bot.send_message(chat_id=update.message.chat_id, text=text)


def draw_contours(image_array, metadata):
    """Отрисовка контуров по bbox и подсчет кол-ва найденных особей"""

    tigers_count = 0
    leos_count = 0

    for bbox in metadata['bbox']:
        class_name = 'Leopard' if bbox['bbox_id'] == 1 else 'Tiger'

        if bbox['bbox_id'] == 1:
            leos_count += 1

        else:
            tigers_count += 1

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

    return leos_count, tigers_count

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
    is_princess = response.json()['is_princess']
    # detected_date = response.json()['date']

    leos_count, tigers_count = draw_contours(image_array, metadata)
    image = transform_pil_image_to_bytes(image_array)

    print('vse ok')

    bot.send_photo(chat_id=update.message.chat_id, photo=image)

    if is_princess == True:
        bot.send_message(chat_id=update.message.chat_id, text=f"На фото обнаружено {leos_count} леопарда(ов)🐆, \n{tigers_count} тигра(ов)🐯.\n Также мы определили, что это наша принцесска👸👑",
                        reply_to_message_id=update.message.message_id,
                        parse_mode=telegram.ParseMode.HTML)
    else:
        bot.send_message(chat_id=update.message.chat_id, text=f"На фото обнаружено {leos_count} леопарда(ов)🐆, \n{tigers_count} тигра(ов)🐯.",
                        reply_to_message_id=update.message.message_id,
                        parse_mode=telegram.ParseMode.HTML)



start_handler = CommandHandler(['start', 'help'], start)
process_image_handler = MessageHandler(Filters.photo | Filters.document, bot_image_processing)

dispatcher.add_handler(start_handler)
dispatcher.add_handler(process_image_handler)

updater.start_polling()
