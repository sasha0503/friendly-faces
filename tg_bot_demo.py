from test_inference import detect

import os
import torch
import telebot
import numpy as np

from dotenv import load_dotenv

load_dotenv()
TELE_API_KEY = os.getenv('TELE_API_KEY')

emb_database = np.load('embeddings_database_new.npy', allow_pickle=True).item()


def find_person(image_path):
    img_probs, img_embedding = detect(image_path)
    img_probs, img_embedding = detect(image_path)
    min_dist = 250
    person = None
    for key in emb_database.keys():
        dist = 0
        for emb in emb_database[key]:
            dist += torch.dist(img_embedding, emb)
        dist /= len(emb_database[key])
        if dist < min_dist:
            min_dist = dist
            person = key

    return person, min_dist


bot = telebot.TeleBot(TELE_API_KEY)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Привіт, надішли мені фото, а я скажу, хто на ньому👀")


@bot.message_handler(content_types=['text', 'document', 'audio', 'video', 'voice', 'sticker'])
def reject_text(message):
    bot.send_message(message.chat.id, "Мені треба фото👀")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        image_path = f'data/{message.photo[-1].file_id}.jpg'
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open(image_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        try:
            person, dist = find_person(image_path)
        except:
            bot.send_message(message.chat.id, "Не бачу тут людей🤨")
            os.remove(image_path)
            return
        translates = {'lyubchyk': 'Любчик', 'vlad': 'Владік', 'slavko': 'Славко', 'sasha': 'Саша', 'bodya': 'Бодя',
                      'stepan': 'Степан', 'pavlo': 'Паша', 'dima': 'Діма'}
        if person is None:
            bot.send_message(message.chat.id, "Я не знаю цю людину🤨")
            os.remove(image_path)
        else:
            person = translates.get(person, person)
            bot.send_message(message.chat.id, person)
            os.remove(image_path)
    except Exception as e:
        bot.send_message(message.chat.id, "Щось пішло не так👀")
        print(e)


if __name__ == '__main__':
    bot.polling()
