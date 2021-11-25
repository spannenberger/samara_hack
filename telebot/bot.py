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
import json

TOKEN = bot_token
URL = 'http://localhost:5000/api/test'

content_type = 'image/jpeg'
headers = {'content-type': content_type}

updater = Updater(token=TOKEN)
dispatcher = updater.dispatcher

print('initializing completed')


def process_image(image_bytes):
    image = np.asarray(bytearray(image_bytes), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def transform_pil_image_to_bytes(image):
    image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)
    return buffer


def get_image_bytes(bot, update):
    print(bot)
    if not update.message.photo:
        file_id = update.message.document['file_id']
    else:
        file_id = update.message.photo[-1]['file_id']
    file = bot.getFile(file_id)
    image = file.download_as_bytearray()
    return image, file_id


def start(bot, update):
    text = 'Привет, я бот распознающий животных в заповедниках, пришли мне фото и я отрисую всех животных, ' \
           'а так же выведу виды этих животных \n'
    bot.send_message(chat_id=update.message.chat_id, text=text)


def bot_image_processing(bot, update):
    image_bytes, file_id = get_image_bytes(bot, update)

    image_array = process_image(image_bytes)
    _, img_encoded = cv2.imencode('.jpg', image_array)
    response = requests.post(URL, data=img_encoded.tostring(), headers=headers)
    metadata = response.json()['image']
    for bbox in metadata['bbox']:
        cv2.rectangle(image_array, (bbox['bbox']['x1'], bbox['bbox']['y1']),\
                      (bbox['bbox']['x2'], bbox['bbox']['y2']), (255, 0, 0), 3)
    image = transform_pil_image_to_bytes(image_array)


    bot.send_photo(chat_id=update.message.chat_id, photo=image)
    bot.send_message(chat_id=update.message.chat_id, text=metadata,
                     reply_to_message_id=update.message.message_id,
                     parse_mode=telegram.ParseMode.HTML)




start_handler = CommandHandler(['start', 'help'], start)
process_image_handler = MessageHandler(Filters.photo | Filters.document, bot_image_processing)

dispatcher.add_handler(start_handler)
dispatcher.add_handler(process_image_handler)

updater.start_polling()
