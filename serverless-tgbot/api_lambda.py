import json
import os
import sys
import boto3
import logging
from tg_client import TgClient
from s3_helper import S3Helper

log = logging.getLogger()
log.setLevel(logging.DEBUG)

BUCKET = os.environ['TG_BOT_S3_BUCKET']
SQS_NST_NAME = os.environ['TG_BOT_SQS_NST_NAME']
SQS_GAN_NAME = os.environ['TG_BOT_SQS_GAN_NAME']

def handler(event, context):
    
    log.debug("logging test")    

    try:
        data = json.loads(event["body"])
        chat_id = data['message']['chat']['id']        

        tg_client = TgClient(chat_id)
        s3 = boto3.client('s3')

        try:
            message_id = data['message']['message_id']
            first_name = data["message"]["chat"]["first_name"]

            instructions = \
                f'Привет, {first_name}!\n' + \
                'Пришли фотографию, и отправь команду /gan для переноса предустановленных стилей.\n' + \
                'Для переноса своего стиля - пришли картинку со стилем, в подписи укажи /style, ' + \
                'после чего отправь команду /nst.'

            message = ''

            if 'text' in data['message']:
                message = data['message']['text']
            elif 'caption' in data['message']:
                message = data['message']['caption']

            # received photo
            if 'photo' in data['message']:
                # get file with the best resolution
                resolutions_num = len(data['message']['photo'])
                file_id = str(data['message']['photo'][resolutions_num-1]['file_id'])

                file_content = tg_client.get_file(file_id)

                file_name = ''

                if '/style' in message:
                    file_name = '{}_style.jpg'.format(chat_id)
                else:
                    file_name = '{}_content.jpg'.format(chat_id)

                s3 \
                    .put_object(
                        Bucket=BUCKET,
                        ACL='public-read',
                        Key=file_name,
                        Body=file_content)   
            
            # received text only             
            else:
                sqs = boto3.client('sqs')

                prefix = '{}_{}'.format(chat_id, message_id)

                if '/gan' in message:
                    if is_content_image_exist(s3, chat_id):                
                        # copying current images to new ones with unique keys
                        copy_content_image(s3, chat_id, prefix)   

                        queue_url = sqs.get_queue_url(QueueName=SQS_GAN_NAME)['QueueUrl']

                        sqs.send_message(
                                QueueUrl=queue_url,
                                MessageBody=json.dumps({ 'prefix': prefix, 'chat_id': chat_id }))

                        tg_client.send_message('GAN - Поставил в очередь на обработку...')
                    else:
                        tg_client.send_message('GAN - Нет фотографии для обработки, подробнее /start')

                elif '/nst' in message:
                    if is_content_image_exist(s3, chat_id) and is_style_image_exist(s3, chat_id):                
                        # copying current images to new ones with unique keys
                        copy_content_image(s3, chat_id, prefix)   
                        copy_style_image(s3, chat_id, prefix)   

                        queue_url = sqs.get_queue_url(QueueName=SQS_NST_NAME)['QueueUrl']

                        sqs.send_message(
                                QueueUrl=queue_url,
                                MessageBody=json.dumps({ 'prefix': prefix, 'chat_id': chat_id }))

                        tg_client.send_message('NST - Поставил в очередь на обработку...')
                    else:
                        tg_client.send_message('NST - Нет фотографии или стиля для обработки, подробнее /start')

                else:
                    tg_client.send_message(instructions)

        except Exception as e:
            tg_client.send_message('Упс, ошибочка - {}'.format(e))

    except Exception as e:
        print(e)

    return {"statusCode": 200}

def is_content_image_exist(s3_client, chat_id):
    return S3Helper.is_file_exist(
        s3_client,
        BUCKET,
        '{}_content.jpg'.format(chat_id))

def is_style_image_exist(s3_client, chat_id):
    return S3Helper.is_file_exist(
        s3_client,
        BUCKET,
        '{}_style.jpg'.format(chat_id))

def copy_content_image(s3_client, chat_id, prefix):
    s3_client \
        .copy_object(
            Bucket=BUCKET,
            CopySource='/{}/{}_content.jpg'.format(BUCKET, chat_id),
            Key='{}_content.jpg'.format(prefix))

def copy_style_image(s3_client, chat_id, prefix):
    s3_client \
        .copy_object(
            Bucket=BUCKET,
            CopySource='/{}/{}_style.jpg'.format(BUCKET, chat_id),
            Key='{}_style.jpg'.format(prefix))