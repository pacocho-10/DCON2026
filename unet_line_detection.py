import os, cv2, numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from arange_data.augmentation import augment_images
layers = tf.keras.layers
models = tf.keras.models
# ======== U-Netの定義 ========
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def build_light_unet(input_shape=(128,128,1)):
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 16); p1 = layers.MaxPooling2D((2,2))(c1)
    c2 = conv_block(p1, 32); p2 = layers.MaxPooling2D((2,2))(c2)
    c3 = conv_block(p2, 64)

    u2 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c3)
    u2 = layers.concatenate([u2, c2])
    c4 = conv_block(u2, 32)

    u1 = layers.Conv2DTranspose(16, 2, strides=2, padding='same')(c4)
    u1 = layers.concatenate([u1, c1])
    c5 = conv_block(u1, 16)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ======== データの読み込み ========
def load_data(img_dir, mask_dir, size=(128,128)):
    imgs, masks = [], []
    for fname in os.listdir(img_dir):
        if not fname.endswith('.png'): continue
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(mask_path): continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size) / 255.0
        mask = cv2.resize(mask, size) / 255.0

        imgs.append(np.expand_dims(img, -1))
        masks.append(np.expand_dims(mask, -1))
    return np.array(imgs, np.float32), np.array(masks, np.float32)

x, y = load_data('dataset/images', 'dataset/masks')
x_aug, y_aug = augment_images(x, y, n=30)
x = np.concatenate([x, x_aug])
y = np.concatenate([y, y_aug])
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

# ======== モデル構築と学習 ========
model = build_light_unet()
model.summary()

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20, batch_size=4
)

# ======== 学習結果の確認 ========
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.show()

# ======== 推論テスト ========
test_img = x_val[0]
pred = model.predict(np.expand_dims(test_img, axis=0))[0,:,:,0]
pred_mask = (pred > 0.5).astype(np.uint8)

plt.subplot(1,3,1); plt.imshow(test_img[:,:,0], cmap='gray'); plt.title('Input')
plt.subplot(1,3,2); plt.imshow(y_val[0][:,:,0], cmap='gray'); plt.title('Ground Truth')
plt.subplot(1,3,3); plt.imshow(pred_mask, cmap='gray'); plt.title('Predicted')
plt.show()

# ======== モデル保存 ========
model.save('light_unet_line_detector.h5')