import random
import cv2
import numpy as np
import albumentations as A
import glob

path_test= "test\images"

def get_aug():
    for iter in  glob.glob(path_test + '/*'):
        if(iter[0] != '.'):
            image = cv2.imread(iter)
            # angle = np.arange(-10,11,1)
            #  angle0 = random.choice(angle)
            # shift = 0.01*np.arange(-10,11,1)
            # shift0 = random.choice(shift)
            # transform = A.ShiftScaleRotate(shift_limit=shift0,rotate_limit=20,scale_limit=0,p=0.5)
            transform = A.HorizontalFlip(p=0.5)
            augmented_image = transform(image=image)['image']
            path = iter.replace('test\images\\','test_gen\\images\\gen_')
            cv2.imwrite(f'{path}', augmented_image)

# Загрузите изображение

# Примените аугментацию с помощью функции get_aug
get_aug()

# Отобразите исходное и аугментированное изображения
# cv2.imwrite('augmented_photo.jpg', augmented_image)
# cv2.imwrite('photo.jpg', image)

# Сохраните аугментированное изображение на диск
# cv2.imwrite('augmented_photo.jpg', augmented_image)