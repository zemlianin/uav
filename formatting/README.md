**Этапы разработки**

***Поиск  и обработка изображений***

Нахождение изображений со следующими классами: легкобронированая техника (БМП, БТР, БРДМ, пикапы, военная автомобильная и специальная техника (бронемобили, командно-штабные машины, транспортированная техника, машины связи, автомобильные комплексы управления), танки, САУ, стационарные объекты военной инфраструктуры (узлы связи, центры управления, стационарные объекты РЭБ, приемо-передающие устройства, позиции боевых порядков, опорные пункты, скопление живой силы противника).

Изображения грузились из Яндекса. Искались похожие изображения по некоторому заранее ранее найденному. Изначально скачивание было вручную, потом был написан скрипт для автоматического скачивания изображений из интернета на ЯП Python с использованием библиотеки Selenium Wire. Код был основан на https://github.com/bobokvsky/yandex-images-download и https://github.com/DimasVeliz/ImagesFromYandex/tree/main. Код доступен в папке ImagesFromYandex.

Большинство изображений содержат нерелевантные объекты и обладают плохим качеством, их стоит отфильтровывать.

Также стоит обратить внимание на разрешение изображений, для YOLOv8 изображения обладают расширением 640х640. Для решения данной проблемы может использоваться динамический padding и динамический cropping (код прилагается: preprocess_new_images.py для разметки до подгрузки в Roboflow, preprocess_dataset.py для разметки изображений с label-ами).

***Разметка***

Для разметки использовался сервис аннотирования Roboflow (датасет (без применения динамических паддингов и cropping-ов) в формате YOLOv8 прилагается: папка 'military vehicles detection.yolov8').
Каждый класс на изображениях отмечался определенным образом в bbox. Так, у каждого вида объектов был свой класс выделенный определенным цветом.
Как организовать датасет: загружаем набор изображений, переходим на вкладку аннотирования, дальше можно посмотреть гайды.

***Обучение модели***

Для MVP решения данной задачи были использованы средства обучения алгоритма внутри Roboflow. Также, внутри данного сервиса данные разделялись на набор обучения, набор валидации и набор тестирования алгоритма.
Качество модели проиллюстрировано на презентации.

***Рекомендации по дальнейшей работе***

Найти большое (примерно 10к+) количество изображений с широким разнообразием классов (примерно по 1.5к+ каждый).
Разметить изображения очень аккуратно, чтобы, по возможности, в bbox присутствовал только объект, при этом границы bbox объединялись с границами объекта.
Предобработать изображения с использованием различных средств из библиотеки cv, желательно рассмотреть такие приемы как нормирование по imagenet, различная работа с каналами изображений, в т.ч изменение геометрических параметров.
Для более качественного обучения модели стоит выгрузить актуальную предобученную SoTA версию модели. Далее работать и улучшать только ее.
Также можно добавить аугментацию.
Дальнейшие действия будут зависеть от функционала внедрения алгоритма в физическую «машину».
