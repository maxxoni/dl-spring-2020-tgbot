import json
import os
import sys
import boto3
from botocore.exceptions import ClientError
import logging
from tg_client import TgClient
from s3_helper import S3Helper

log = logging.getLogger()
log.setLevel(logging.DEBUG)

here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here, "./vendored"))

import requests

BUCKET = os.environ['TG_BOT_S3_BUCKET']
EC2_NST_SERVER_IP = os.environ['EC2_NST_SERVER_IP']

def handler(event, context):
    for record in event['Records']:
        message = json.loads(record["body"])
        chat_id = str(message['chat_id'])
        prefix = str(message['prefix'])

        tg_client = TgClient(chat_id)
        s3 = boto3.client('s3')

        # do not process items with same prefix if
        # already done - SQS could send duplicates
        output_exists = S3Helper.is_file_exist(
            s3, 
            BUCKET,
            '{}_output.jpg'.format(prefix))

        if(output_exists is False):
            try:
                tg_client.send_message('NST - Запустил обработку...')

                # requesting EC2 instance to process the image
                # NST content and style images
                
                url = 'http://{}/api/?img_prefix={}'.format(EC2_NST_SERVER_IP, prefix)
                response = requests.get(url)

                responseJson = json.loads(response.text)
                file_name = responseJson['file_name']     
                file_url = 'https://{}.s3.amazonaws.com/{}'.format(BUCKET, file_name)

                # sending output image back
                tg_client.send_photo(file_url)
            except Exception as e:
                tg_client.send_message('NST - Упс, ошибочка - {}'.format(e))