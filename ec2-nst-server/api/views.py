from django.shortcuts import render
from django.http import HttpResponse
import json
import os, pwd
import sys

here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here, "./vendored"))

import boto3
import requests

uploads_dir = '/home/maxim_vorotynets/uploads/'

# Create your views here.
def index(request):
    response = ''

    try:
        img_prefix = request.GET['img_prefix']

        s3 = boto3.client('s3')

        bucket = 'dl-spring-2020-tgbot-s3-bucket'

        style_img_file_name = '{}_style.jpg'.format(img_prefix)
        content_img_file_name = '{}_content.jpg'.format(img_prefix)

        style_img_file_path = '{}{}'.format(uploads_dir, style_img_file_name)
        content_img_file_path = '{}{}'.format(uploads_dir, content_img_file_name)

        s3.download_file(bucket, style_img_file_name, style_img_file_path)
        s3.download_file(bucket, content_img_file_name, content_img_file_path)

        output_image_file_name = nst(style_img_file_path, content_img_file_path, img_prefix)

        with open('{}{}'.format(uploads_dir, output_image_file_name), 'rb') as output_data:
            s3 \
                .put_object(
                    Bucket=bucket,
                    ACL='public-read',
                    Key=output_image_file_name,
                    Body=output_data)

        response = json.dumps({ 'result': 'success', 'file_name': output_image_file_name })
    except Exception as e:
        response = json.dumps({ 'result': 'error', 'message': e })

    return HttpResponse(response)

###################################
# almost copy/paste from my HW 19 #
###################################

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import copy

import gc

def nst(style_img_file_path, content_img_file_path, img_prefix):
    def clean_memory():
        gc.collect()
        torch.cuda.empty_cache()

    clean_memory()
    
    try:
        # maximum size of the squared image
        imsize = 384

        num_epochs = 80
        style_weight = 100000
        content_weight = 1
        scheduler_step_size = 10
        scheduler_gamma = 0.3

        # вписывает картинку в квадрат, пустоты заполняет нулями
        def make_square(im, size, fill_color=(0, 0, 0, 0)):
            x, y = im.size
            size = max(size, x, y)
            new_im = Image.new('RGB', (size, size), fill_color)
            new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
            return new_im

        # пытаемся работать на GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # преобразователь картинок для ввода в нейронную сеть
        loader = transforms.Compose([
            # увеличиваем изображение до нужного размера
            transforms.Resize(imsize),  
            transforms.CenterCrop(imsize),
            # превращаем в тензор
            transforms.ToTensor()]) 

        # загружает картинку, преобразовывает ее для работы с нейронкой,
        # возвращает маску картинки если требуется (нужно, чтобы "пустоты"
        # занулялись на каждой эпохе style transfer и не искажали картинку на входе)
        def image_loader(image_name, return_mask=False):
            image = Image.open(image_name)
            image_size = image.size    
            image = make_square(image, size=imsize)
            image = loader(image).unsqueeze(0)

            if(not return_mask):
                return image.to(device, torch.float)
            else:
                mask_image = Image.new('RGB', image_size, (255, 255, 255))
                mask_image = make_square(mask_image, size=imsize)
                mask_image = loader(mask_image).unsqueeze(0)

                return image.to(device, torch.float), mask_image.to(device, torch.float)

        # заргужаем исходную картинку и ее маску
        content_img, content_mask_img = image_loader(content_img_file_path, return_mask=True)
        style_img = image_loader(style_img_file_path)

        # заргужаем картинки стилей в массив
        style_imgs = [
            style_img,          
        ]

        # загружаем маски в массив, эти маски будут обозначать на какую область исходной картинки
        # какой стиль будет переноситься (в том же порядке как и в массиве стилей)
        mask_imgs = [    
            content_mask_img,
        ]

        class ContentLoss(nn.Module):

            def __init__(self, target):
                super(ContentLoss, self).__init__()
                # we 'detach' the target content from the tree used
                # to dynamically compute the gradient: this is a stated value,
                # not a variable. Otherwise the forward method of the criterion
                # will throw an error.
                self.target = target.detach() # это константа. Убираем ее из дерева вычислений
                self.loss = F.mse_loss(self.target, self.target) # to initialize with something

            def forward(self, input):
                self.loss = F.mse_loss(input, self.target)
                return input

        def gram_matrix(input):
            batch_size , h, w, f_map_num = input.size()  # batch size(=1)
            # b=number of feature maps
            # (h,w)=dimensions of a feature map (N=h*w)

            features = input.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL

            G = torch.mm(features, features.t())  # compute the gram product

            # we 'normalize' the values of the gram matrix
            # by dividing by the number of element in each feature maps.
            return G.div(batch_size * h * w * f_map_num)

        class StyleLoss(nn.Module):
        
            def __init__(self, target_feature, mask):
                super(StyleLoss, self).__init__()
                
                # ресайзим маску к размеру target_feature и
                # клонируем ее по количеству каналов
                self.mask = mask.cpu().clone().squeeze()
                stacker = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(target_feature.shape[2]),  
                        transforms.ToTensor()])
                self.mask = stacker(self.mask)
                self.mask = torch.stack([self.mask[0]], dim=0)
                self.mask = self.mask.repeat(target_feature.shape[1], 1, 1)
                self.mask = torch.stack([self.mask], dim=0)
                self.mask = self.mask.to(device)
                
                self.target = gram_matrix(target_feature * self.mask).detach()
                self.loss = F.mse_loss(self.target, self.target)

            def forward(self, input):
                G = gram_matrix(input * self.mask)
                self.loss = F.mse_loss(G, self.target)
                return input

        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        class Normalization(nn.Module):
        
            def __init__(self, mean, std):
                super(Normalization, self).__init__()
                # .view the mean and std to make them [C x 1 x 1] so that they can
                # directly work with image Tensor of shape [B x C x H x W].
                # B is batch size. C is number of channels. H is height and W is width.
                self.mean = torch.tensor(mean).view(-1, 1, 1)
                self.std = torch.tensor(std).view(-1, 1, 1)

            def forward(self, img):
                # normalize img
                return (img - self.mean) / self.std

        content_layers_default = ['conv_4']
        style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        os.environ['TORCH_HOME'] = '{}torch'.format(uploads_dir)

        cnn = models.vgg19(pretrained=True).features.to(device).eval()

        def get_style_model_and_losses(
            cnn,
            normalization_mean, 
            normalization_std,
            content_img,
            style_imgs,    
            mask_imgs,
            content_layers=content_layers_default,
            style_layers=style_layers_default):

            cnn = copy.deepcopy(cnn)

            # первым идет слой нормализации
            normalization = Normalization(normalization_mean, normalization_std).to(device)

            # just in order to have an iterable access to or list of content/syle
            # losses
            content_losses = []
            style_losses = []

            # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
            # to put in modules that are supposed to be activated sequentially
            model = nn.Sequential(normalization)

            i = 0  # increment every time we see a conv
            for layer in cnn.children():
                if isinstance(layer, nn.Conv2d):
                    i += 1
                    name = 'conv_{}'.format(i)
                elif isinstance(layer, nn.ReLU):
                    name = 'relu_{}'.format(i)
                    # The in-place version doesn't play very nicely with the ContentLoss
                    # and StyleLoss we insert below. So we replace with out-of-place
                    # ones here.
                    #Переопределим relu уровень
                    layer = nn.ReLU(inplace=False)
                elif isinstance(layer, nn.MaxPool2d):
                    name = 'pool_{}'.format(i)
                elif isinstance(layer, nn.BatchNorm2d):
                    name = 'bn_{}'.format(i)
                else:
                    raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

                model.add_module(name, layer)

                if name in content_layers:
                    # add content losses            
                    content_target = model(content_img).detach()
                    content_loss = ContentLoss(content_target)
                    model.add_module("content_loss_{}".format(i), content_loss)
                    content_losses.append(content_loss)

                if name in style_layers:
                    # добавляем лосс функцию для каждого стиля со своей маской
                    for s_i in range(len(style_imgs)):
                        target_feature = model(style_imgs[s_i]).detach()
                        style_loss = StyleLoss(target_feature, mask_imgs[s_i])
                        model.add_module("style_loss_{}_{}".format(s_i, i), style_loss)
                        style_losses.append(style_loss)

            # now we trim off the layers after the last content and style losses
            # выбрасываем все уровни после последнего styel loss или content loss
            for i in range(len(model) - 1, -1, -1):
                if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                    break

            model = model[:(i + 1)]

            return model, content_losses, style_losses

        def get_input_optimizer_and_scheduler(input_img):
            # this line to show that input is a parameter that requires a gradient
            # добавляет содержимое тензора катринки в список изменяемых оптимизатором параметров
            optimizer = optim.LBFGS([input_img.requires_grad_()]) 
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
            return optimizer, scheduler

        def run_style_transfer(
            cnn, 
            normalization_mean, 
            normalization_std,
            content_img, 
            content_mask_img, 
            style_imgs, 
            mask_imgs,
            input_img, 
            num_epochs,
            style_weight, 
            content_weight):

            """Run the style transfer."""
            print('Building the style transfer model..')
            model, content_losses, style_losses = \
                get_style_model_and_losses(
                    cnn,
                    normalization_mean, 
                    normalization_std, 
                    content_img, 
                    style_imgs, 
                    mask_imgs)
            
            optimizer, scheduler = get_input_optimizer_and_scheduler(input_img)

            print('Optimizing..')
            
            for epoch in range(num_epochs):
                
                def closure():
                    
                    # correct the values 
                    # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                    input_img.data.clamp_(0, 1)

                    optimizer.zero_grad()
            
                    model(input_img)

                    style_score = 0
                    content_score = 0

                    for sl in style_losses:
                        style_score += sl.loss
                    for cl in content_losses:
                        content_score += cl.loss
                        
                    style_score *= style_weight
                    content_score *= content_weight

                    loss = style_score + content_score
                    loss.backward()

                    # умножаем градиент на маску изображения, чтобы пустоты не мешали
                    input_img.grad *= content_mask_img

                    if (epoch + 1) % 10 == 0:
                        print("epoch {}:".format(epoch + 1))
                        print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                                style_score.item(), content_score.item()))
                        print()

                    return style_score + content_score

                optimizer.step(closure)

                scheduler.step()

            # a last correction...
            input_img.data.clamp_(0, 1)

            return input_img

        clean_memory()

        input_img = content_img.clone()

        output = run_style_transfer(
            cnn, 
            cnn_normalization_mean, 
            cnn_normalization_std,
            content_img, 
            content_mask_img,
            style_imgs,  
            mask_imgs, 
            input_img, 
            num_epochs=num_epochs,
            style_weight=style_weight, 
            content_weight=content_weight)

        unloader = transforms.ToPILImage() 

        image = unloader(output.detach().cpu().squeeze())

        output_image_file_name = '{}_output.jpg'.format(img_prefix)

        image.save('{}{}'.format(uploads_dir, output_image_file_name))

        clean_memory()

        return output_image_file_name
    except Exception as e:
        clean_memory()
        raise e