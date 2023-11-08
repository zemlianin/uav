import random
import time

import cv2
from matplotlib import pyplot as plt

import albumentations as A

# функция визуализации
def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


# ввод изображения
image = cv2.imread('data/images/image_1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
visualize(image)


# ввод лейблов
with open('data/labels/label_1.txt') as f:
    lines = f.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()
    lines[i][0], lines[i][-1] = lines[i][-1], lines[i][0]
    for j in range(len(lines[i]) - 1):
        lines[i][j] = float(lines[i][j])
bboxes = lines
print(bboxes)


# аугментация изображения


# транспозиция изображения
def trans(image, bboxes):
    transform = A.Transpose(p=1)
    random.seed(1) # сид не меняем, иначе аутпут пикчи будет рандомный (даже без этой строчки)
    transformed = transform(image=image, bboxes=bboxes)
    augmented_image = transformed['image'] # трансформация
    #transformed_bboxes = transformed['bboxes']
    visualize(augmented_image) # вывод пикчи
    #print(transformed_bboxes)


# горизонтальная трансформация изображения
def hor_flip(image):
    transform = A.HorizontalFlip(p=1)
    random.seed(1) # сид не меняем, иначе аутпут пикчи будет рандомный (даже без этой строчки)
    augmented_image = transform(image=image)['image'] # трансформация
    visualize(augmented_image) # вывод пикчи


# вертикальная трансформация изображения
def ver_flip(image):
    transform = A.VerticalFlip(p=1)
    random.seed(1) # сид не меняем, иначе аутпут пикчи будет рандомный (даже без этой строчки)
    augmented_image = transform(image=image)['image'] # трансформация
    visualize(augmented_image) # вывод пикчи


# x - y зеркальная трансформация изображения
def trans180(image):
    transform = A.Compose([
        A.Transpose(p=1),
        A.ShiftScaleRotate(p=1, rotate_limit=[179, 181]),
    ])
    random.seed(1) # сид не меняем, иначе аутпут пикчи будет рандомный (даже без этой строчки)
    augmented_image = transform(image=image)['image'] # трансформация
    visualize(augmented_image) # вывод пикчи


# 90 поворотная трансформация изображения
def rotate90(image):
    transform = A.ShiftScaleRotate(p=1, rotate_limit=[89, 91])
    random.seed(1) # сид не меняем, иначе аутпут пикчи будет рандомный (даже без этой строчки)
    augmented_image = transform(image=image)['image'] # трансформация
    visualize(augmented_image) # вывод пикчи


# -90 поворотная трансформация изображения
def rotate90m(image):
    transform = A.ShiftScaleRotate(p=1, rotate_limit=[-89, -91])
    random.seed(1) # сид не меняем, иначе аутпут пикчи будет рандомный (даже без этой строчки)
    augmented_image = transform(image=image)['image'] # трансформация
    visualize(augmented_image) # вывод пикчи


# 180 поворотная трансформация изображения
def rotate180(image):
    transform = A.ShiftScaleRotate(p=1, rotate_limit=[179, 181])
    random.seed(1) # сид не меняем, иначе аутпут пикчи будет рандомный (даже без этой строчки)
    augmented_image = transform(image=image)['image'] # трансформация
    visualize(augmented_image) # вывод пикчи


#trans(image)