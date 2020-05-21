import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from matplotlib import pyplot as plt
import numpy as np
import os
import random
from keras.preprocessing.image import load_img, img_to_array

if __name__ == '__main__':
    # Import noisy office documents dataset
    noisy_imgs_path = 'Noisy_Documents/noisy/'
    clean_imgs_path = 'Noisy_Documents/clean/'

    X_train_noisy = []
    X_train_clean = []

    for file in sorted(os.listdir(noisy_imgs_path)):
      img = load_img(noisy_imgs_path+file, color_mode='grayscale', target_size=(420,540))
      img = img_to_array(img).astype('float32')/255
      X_train_noisy.append(img)

    for file in sorted(os.listdir(clean_imgs_path)):
      img = load_img(clean_imgs_path+file, color_mode='grayscale', target_size=(420,540))
      img = img_to_array(img).astype('float32')/255
      X_train_clean.append(img)

    # 넘파이 배열로 변환한다
    X_train_noisy = np.array(X_train_noisy)
    X_train_clean = np.array(X_train_clean)

    # 노이즈 이미지 첫 20개를 테스트 이미지로 사용한다
    X_test_noisy = X_train_noisy[0:20,]
    X_train_noisy = X_train_noisy[21:,]

    # 정상 이미지 첫 20개를 테스트 이미지로 사용한다
    X_test_clean = X_train_clean[0:20,]
    X_train_clean = X_train_clean[21:,]

    # 모델을 만들고 훈련시킨다
    conv_autoencoder = Sequential()
    conv_autoencoder.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(420,540,1), activation='relu', padding='same'))
    conv_autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'))
    conv_autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
    conv_autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
    conv_autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'))
    conv_autoencoder.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
    conv_autoencoder.add(Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same'))

    conv_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    conv_autoencoder.fit(X_train_noisy, X_train_clean, epochs=10)

    output = conv_autoencoder.predict(X_test_noisy)

    # 결과 출력
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)

    randomly_selected_imgs = random.sample(range(X_test_noisy.shape[0]),2)

    for i, ax in enumerate([ax1, ax4]):
        idx = randomly_selected_imgs[i]
        ax.imshow(X_test_noisy[idx].reshape(420,540), cmap='gray')
        if i == 0:
            ax.set_title("Noisy Images")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    for i, ax in enumerate([ax2, ax5]):
        idx = randomly_selected_imgs[i]
        ax.imshow(X_test_clean[idx].reshape(420,540), cmap='gray')
        if i == 0:
            ax.set_title("Clean Images")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    for i, ax in enumerate([ax3, ax6]):
        idx = randomly_selected_imgs[i]
        ax.imshow(output[idx].reshape(420,540), cmap='gray')
        if i == 0:
            ax.set_title("Output Denoised Images")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
