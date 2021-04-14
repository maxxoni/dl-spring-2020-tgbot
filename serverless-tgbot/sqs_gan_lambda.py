# this import statement is needed if you want to use the AWS Lambda Layer called "pytorch-v1-py36"
# it unzips all of the pytorch & dependency packages when the script is loaded to avoid the 250 MB unpacked limit in AWS Lambda
try:
    import unzip_requirements
except ImportError:
    pass

import json
import os
import sys
import boto3
from botocore.exceptions import ClientError
import logging
from io import BytesIO
import re
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tg_client import TgClient
from s3_helper import S3Helper

log = logging.getLogger()
log.setLevel(logging.DEBUG)

BUCKET = os.environ['TG_BOT_S3_BUCKET']
GAN_STYLES_BUCKET = os.environ['TG_BOT_S3_GAN_STYLES_BUCKET']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def handler(event, context):
    s3 = boto3.client('s3')

    imsize = 512

    content_transform = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    for record in event['Records']:
        message = json.loads(record["body"])
        chat_id = str(message['chat_id'])
        prefix = str(message['prefix'])

        tg_client = TgClient(chat_id)

        style_models = s3.list_objects(Bucket=GAN_STYLES_BUCKET)

        styles = []

        for style in style_models['Contents']:
            if '.pth' in style['Key']:
                styles.append(style['Key'])

        # do not process items with same prefix if
        # already done - SQS could send duplicates
        output_exists = S3Helper.is_file_exist(
            s3, 
            BUCKET,
            '{}_output_{}.jpg'.format(prefix, styles[0]))

        if(output_exists is False):
            try:
                tg_client.send_message('GAN - Нашел {} стиля, поехали!'.format(len(styles)))

                for i in range(len(styles)):
                    style = styles[i]

                    message_text = 'GAN - Запустил обработку стилем {}...'.format(style)

                    if i == len(styles)-1:
                        message_text = 'GAN - И последний - {}...'.format(style)
                    
                    tg_client.send_message(message_text)                    

                    content_img_file_name = '{}_content.jpg'.format(prefix)
                    content_img_file_path = f"/tmp/{content_img_file_name}"

                    s3.download_file(
                        BUCKET, 
                        content_img_file_name, 
                        content_img_file_path)

                    img = Image.open(content_img_file_path).convert('RGB')
                    img = content_transform(img)
                    img = img.unsqueeze(0).to(device)
                    
                    model_file_name = style
                    model_file_path = f"/tmp/{model_file_name}"

                    s3.download_file(
                        GAN_STYLES_BUCKET,
                        model_file_name, 
                        model_file_path)

                    style_model = TransformerNet()
                    state_dict = torch.load(model_file_path, map_location=torch.device('cpu'))
                    
                    for k in list(state_dict.keys()):
                        if re.search(r'in\d+\.running_(mean|var)$', k):
                            del state_dict[k]
                    style_model.load_state_dict(state_dict)
                    style_model.to(device)
                    
                    with torch.no_grad():
                        output = style_model(img)

                    img = output[0].clone().clamp(0, 255).numpy()
                    img = img.transpose(1, 2, 0).astype("uint8")
                    print(img.shape, type(img))                                
                    
                    img = Image.fromarray(img)
                    
                    output_image_file_name = '{}_output_{}.jpg'.format(prefix, style)
                    output_image_file_path = f"/tmp/{output_image_file_name}"

                    img.save(output_image_file_path, format="PNG")

                    with open(output_image_file_path, 'rb') as output_data:
                        s3 \
                            .put_object(
                                Bucket=BUCKET,
                                ACL='public-read',
                                Key=output_image_file_name,
                                Body=output_data)
                    
                    file_url = 'https://{}.s3.amazonaws.com/{}'.format(BUCKET, output_image_file_name)

                    # sending output image back
                    tg_client.send_photo(file_url)

            except Exception as e:
                tg_client.send_message('GAN - Упс, ошибочка - {}'.format(e))

class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out