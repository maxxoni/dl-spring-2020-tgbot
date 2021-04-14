import json
import os
import sys
import logging

log = logging.getLogger()
log.setLevel(logging.DEBUG)

here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here, './vendored'))

import requests

TOKEN = os.environ['TELEGRAM_TOKEN']

BASE_URL = f'https://api.telegram.org/bot{TOKEN}'
BASE_FILE_URL = f'https://api.telegram.org/file/bot{TOKEN}'

class TgClient:
    """
    Telegram Api client
    """

    def __init__(self, chat_id):
        self.chat_id = chat_id

    def send_message(self, text):
        data = { 'text': text.encode('utf8'), 'chat_id': self.chat_id }
        url = f'{BASE_URL}/sendMessage'
        requests.post(url, data)

    def send_photo(self, image_url):
        data = { 'photo': image_url, 'chat_id': self.chat_id }
        url = BASE_URL + "/sendPhoto"
        requests.post(url, data)

    def get_file_url(self, file_id):
        url = BASE_URL + '/getFile?file_id={}'.format(file_id)
        response = requests.get(url)

        file_path = (json.loads(response.text))['result']['file_path']

        return f'{BASE_FILE_URL}/{file_path}'

    def get_file(self, file_id):
        file_url = self.get_file_url(file_id)

        file = requests.get(file_url)

        return file.content