from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np

# 拡張設定
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

def augment_images(x, y, n=30):
    aug_x, aug_y = [], []
    for i in range(len(x)):
        xi = np.expand_dims(x[i], 0)
        yi = np.expand_dims(y[i], 0)
        itx = datagen.flow(xi, batch_size=1, seed=1)
        ity = datagen.flow(yi, batch_size=1, seed=1)
        for _ in range(n):
            aug_x.append(next(itx)[0])
            aug_y.append(next(ity)[0])
    return np.array(aug_x), np.array(aug_y)