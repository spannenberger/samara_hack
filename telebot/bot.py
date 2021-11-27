import telegram
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters
from telegram import InputMediaPhoto
from telegram.ext import Updater
from credentials import bot_token
import cv2
import requests
import os 
import glob

from bot_commands import start, cut_video
from utils import process_image, transform_pil_image_to_bytes, get_video, get_image_bytes, draw_contours


TOKEN = bot_token
URL = 'http://10.10.67.145:5010/api/test'

content_type = 'image/jpeg'
headers = {'content-type': content_type}

updater = Updater(token=TOKEN)
dispatcher = updater.dispatcher

print('initializing completed')


def bot_video_preprocessing(bot, update):
    """Функция для определения животных на видеофрагменте
    
    Открываем полученный видеофрагмент для покадрового предикта модели 

    Args:
        bot: bot - тело нашего бота
        update: command - параметр для обновления данных в боте
    """
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
    while cap.isOpened(): # открываем полученный видеофрагмент

        _, img_frame = cap.read()
        if img_frame is None:
            break

        photo_count += 1
        image_array = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)

        text = 'О.... Кажется, нашел один.... Еще секундочку\n'
        bot.send_message(chat_id=update.message.chat_id, text=text)

        _, img_encoded = cv2.imencode('.jpg', image_array)
        data = img_encoded.tostring()

        response = requests.post(URL, data=data, headers=headers) # отправляем запрос на бэкенд для предикта
        metadata = response.json()['image'] # обрабатываем полученный ответ с результатами работы модели
        
        #@TODO вывод сообщения-отчета о результатах детекции
        leos_count, tigers_count, other_animal_count, is_princess = draw_contours(image_array, metadata).values()
        
        # сохраняем изображение с видеофрагмента
        cv2.imwrite(f'/workspace/bot/photosfrom_video/photo {photo_count}.jpg', cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

        count += 120 # 1 секунда = 30; 2 секунды = 60 и тд - параметр для интервала
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)

    text = 'Обработку закончил, все в норме! Сейчас отправлю альбом фотографий)\n'
    bot.send_message(chat_id=update.message.chat_id, text=text)

    # Формируем альбом фотографий после детекции
    media_group = []
    text = 'some caption for album'
    for img in glob.glob("/workspace/bot/photosfrom_video/*.jpg"):
        media_group.append(InputMediaPhoto(open(img, 'rb')))

    bot.send_media_group(chat_id=update.message.chat_id, media = media_group)


def bot_image_processing(bot, update):
    """Функция обработки запроса с изображением
    
    Получаем и обрабатываем запрос с бэка, формируем сообщение в бота

    Args:
        bot: bot - тело нашего бота
        update: command - параметр для обновления данных в боте
    """

    image_bytes, _ = get_image_bytes(bot, update)

    image_array = process_image(image_bytes)
    _, img_encoded = cv2.imencode('.jpg', image_array)

    data = img_encoded.tostring()

    response = requests.post(URL, data=data, headers=headers) # отправляем запрос на бэк
    metadata = response.json()['image'] # обрабатываем полученный ответ с результатами работы модели

    leos_count, tigers_count, other_animal_count, is_princess = draw_contours(image_array, metadata).values()
    image = transform_pil_image_to_bytes(image_array)

    print('vse ok')
    # проверяем нашли ли мы принцессу
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
