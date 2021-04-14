# README #

Данный репозиторий содержит итоговый проект по курсу "Глубокое Обучение" МФТИ, весенний семестр 2020.

### Описание ###

Заданием на итоговый проект было создание Telegram-бота, который применял бы Neural Style Transfer к присланным фотографиям.   
Обработка должна производиться либо медленным NST алгоритмом (в этом случае также необходимо принять картинку со стилем),
либо Fast NST с уже предобученной сетью GAN моделью.

Репозиторий содержит 2 проекта:

* **serverless-tgbot** - содержит serverless часть для AWS
* **ec2-nst-server** - содержит Django проект для установки на виртуальную машину

Ниже приведена схема работы решения, синими стрелками обозначена обработка запроса, зелеными - ответ.   
Бот работает в полностью асинхронном режиме.

![Схема](https://bitbucket.org/vormax2002/dl-spring-2020-tgbot/raw/e616ef1ec921853ba5aa33c956822bdb5705c9eb/README-FILES/Scheme.png)

* **Telegram** вызывает Webhook (в качестве которого выступает **API Gateway**), куда передает сообщение от пользователя.
* **API Gateway** перенаправляет вызов к **API Lambda**, которая либо обрабатывает запрос, либо возвращает пользователю инструкцию по работе с ботом:

*Привет, {username}!   
Пришли фотографию, и отправь команду /gan для переноса предустановленных стилей.   
Для переноса своего стиля - пришли картинку со стилем, в подписи укажи /style, после чего отправь команду /nst.*

* В зависимости от запроса, **API Lambda** либо сохраняет полученную фотографию/картинку со стилем в **Content S3**, либо инициирует перенос стиля:
    * В случае NST добавляет сообщение в **NST SQS** - дальнейший процесс описан в **NST**
    * В случае GAN добавляет сообщение в **GAN SQS** - дальнейший процесс описан в **GAN**

#### NST ####
* **NST SQS Lambda** получает сообщение из **NST SQS**, и делает HTTP запрос к **NST EC2, Django** для переноса стиля
* **NST EC2, Django** принимает запрос, скачивает из **Content S3** соответствующие запросу content и style картинки, производит перенос стиля, закачивает результат в **Content S3**, возвращает название файла в качестве ответа на запрос
* Получив ответ, **NST SQS Lambda** через **Telegram API** передает URL картинки/результата обработки

#### GAN ####
* **GAN SQS Lambda** получает сообщение из **GAN SQS**, скачивает из **Content S3** соответствующую content картинку, скачивает предобученные модели из **GAN Styles S3**, применяет поочередно каждую модель к content картинке.
* Через **Telegram API** передает URL картинок/результатов обработки

### Технологии ###

#### serverless-tgbot ####
* Использован фреймворк **serverless** - https://www.serverless.com/
* Для синхронизации файлов с предобученными моделями в **GAN Styles S3**, использован plugin **serverless-s3-sync** - https://www.serverless.com/plugins/serverless-s3-sync
* Для использования pytorch в **GAN SQS Lambda**, использован подход и заимствован код из статьи - http://francescopochetti.com/fast-neural-style-transfer-deploying-pytorch-models-to-aws-lambda/ .   
Для обхода ограничения по памяти в Lambda Function, автор, в свою очередь, использует готовый PyTorch Lambda Layer - https://github.com/mattmcclean/sam-pytorch-example , и заимствует код модели из официального репозитория примеров pytorch - https://github.com/pytorch/examples/tree/master/fast_neural_style (откуда я заимствовал предобученные модели)

#### ec2-nst-server ####
* В качестве веб-сервера используется Apache, в качестве фреймворка - Django
* Для переноса стиля я использовал свой код из ноутбука к домашнему заданию 19

### Установка ###

Для начала, нужно в переменных окружения операционной системы экспортировать необходимые настройки (~/.zprofile в случае MacOS):   
   
export AWS_ACCESS_KEY_ID={Ваш **AWS_ACCESS_KEY_ID**}   
export AWS_SECRET_ACCESS_KEY={Ваш **AWS_SECRET_ACCESS_KEY**}   
export TELEGRAM_TOKEN="{Ваш **TELEGRAM_TOKEN**}"   
export EC2_NST_SERVER_IP="{IP адрес Вашего **NST EC2, Django** сервера}"   
export AWS_ACCOUNT="{Ваш **AWS_ACCOUNT**}"   

Далее необходимо поменять названия S3 Buckets, так как они должны быть уникальными:   
   
* **serverless-tgbot/serverless.yml**, поменять названия в строчках:   
   
bucket: {Ваше название **Content S3** bucket}   
gan-styles-bucket: {Ваше название **GAN Style S3** bucket}
   
* **ec2-nst-server/api/views.py**, поменять названия в строчке:   
   
bucket = '{Ваше название **Content S3** bucket}'   

#### serverless-tgbot ####
Здесь все просто - зайти в терминале в папку **serverless-tgbot** и выполнить команду **serverless deploy**

#### ec2-nst-server ####
Выполнить инструкции - https://medium.com/saarthi-ai/ec2apachedjango-838e3f6014ab   
При необходимости доустановить необходимые пакеты (torch, torchvision, boto3)   
Сконфигурировать доступ к сервисам AWS - https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html

### С кем связаться? ###

* write to vormax2002@gmail.com