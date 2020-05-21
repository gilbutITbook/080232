import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
import random

def create_basic_autoencoder(hidden_layer_size):
  model = Sequential() 
  model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
  model.add(Dense(units=784, activation='sigmoid'))
  return model

if __name__ == '__main__':
  # MNIST 데이터셋을 가져온다
  training_set, testing_set = mnist.load_data()
  X_train, y_train = training_set
  X_test, y_test = testing_set

  # 신경망에 사용할 수 있게 벡터 형태를 바꾼다
  X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
  X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

  # 0부터 255까지의 값을 0과 1 사이 값으로 정규화한다
  X_train_reshaped = X_train_reshaped/255.
  X_test_reshaped = X_test_reshaped/255.

  # MNIST 데이터셋에 노이즈를 추가한다
  X_train_noised = X_train_reshaped + np.random.normal(0, 0.5, size=X_train_reshaped.shape) 
  X_test_noised = X_test_reshaped + np.random.normal(0, 0.5, size=X_test_reshaped.shape)
  X_train_noised = np.clip(X_train_noised, a_min=0, a_max=1)
  X_test_noised = np.clip(X_test_noised, a_min=0, a_max=1)

  # 모델을 만들고 훈련시킨다
  basic_denoise_autoencoder = create_basic_autoencoder(hidden_layer_size=16)
  basic_denoise_autoencoder.compile(optimizer='adam', loss='mean_squared_error')
  basic_denoise_autoencoder.fit(X_train_noised, X_train_reshaped, epochs=10)

  output = basic_denoise_autoencoder.predict(X_test_noised)

  # 결과 출력
  fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3, 5)
  randomly_selected_imgs = random.sample(range(output.shape[0]),5)

  # 첫 번째 줄에는 원본 이미지를 그린다
  for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(X_test_reshaped[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
    if i == 0:
      ax.set_ylabel("Original \n Images")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

  # 두 번째 줄에는 노이즈를 추가한 이미지를 그린다
  for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(X_test_noised[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
    if i == 0:
      ax.set_ylabel("Input With \n Noise Added")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

  # 세 번째 줄에는 오토인코더가 출력한 결과 이미지를 그린다
  for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(output[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
    if i == 0:
      ax.set_ylabel("Denoised \n Output")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

  plt.show()
