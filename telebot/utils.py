import numpy as np
import cv2
import io
from PIL import Image

def process_image(image_bytes):
    """Перевод изображения из байтов в необходимый для вывода формат
    
    Args: 
        image_bytes: bytearray - битовое представление изображения
    Return:
        image: cv2.cvtColor - само изображение
    """

    image = np.asarray(bytearray(image_bytes), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def transform_pil_image_to_bytes(image):
    """Перевод из array в изображение

    @TODO DOCS
    Args:
        image: type - представление фотографии
    Return:
        buffer: type - сохраненная фотография
    """

    image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)

    return buffer


def get_image_bytes(bot, update):
    """ Выгрузка полученной фотографии

    Args:
        bot: bot - тело нашего бота
        update: command - параметр для обновления данных в боте
    Return:
        image: bytearray - битовое представление полученного изображения
        file_id: int - уникальный id полученного изображения 
    """

    if not update.message.photo:
        file_id = update.message.document['file_id']
    else:
        file_id = update.message.photo[-1]['file_id']
    file = bot.getFile(file_id)
    image = file.download_as_bytearray()
    return image, file_id


def get_video(bot, update):
    """ Выгрузка полученного видеофрагмента

    Args:
        bot: bot - тело нашего бота
        update: command - параметр для обновления данных в боте
    Return:
        file_path: str - путь до полученного видеофрагмента
    """

    if not update.message.video:
        file_id = update.message.document['file_id']
    else:
        file_id = update.message.video['file_id']
    file = bot.getFile(file_id)
    file_path = file.file_path
    return file_path


def draw_contours(image_array, metadata):
    """ Функция отрисовки контура и подсчета кол-во особей

    Обрабатываем результат работы моделей, извлекая полученный класс животного

    Args:
        image_array: arr - массив-представление изображения
        metadata: - словарь, содержащий ответ работы моделей

    Return:
        counter_dict: dict - словарь с кол-вом определенных животных
    """

    counter_dict = {'leopard': 0, 'tigers': 0, 'other animal': 0, 'is_princess': False}

    for bbox in metadata['bbox']:
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