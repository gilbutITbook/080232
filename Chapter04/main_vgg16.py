import os
import random
import warnings
warnings.filterwarnings("ignore")
from utils import train_test_split

src = 'Dataset/PetImages/'

# 데이터셋이 있는지 먼저 확인한다. 없다면 내려 받아야 한다
if not os.path.isdir(src):
    print("""
          Dataset not found in your computer.
          Please follow the instructions in the link below to download the dataset:
          https://raw.githubusercontent.com/PacktPublishing/Neural-Network-Projects-with-Python/master/chapter4/how_to_download_the_dataset.txt
          """)
    quit()


# train/test 폴더가 없다면 생성한다
if not os.path.isdir(src+'train/'):
	train_test_split(src)

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator

# 초매개변수 정의
INPUT_SIZE = 128 # 코드를 실행하는 데 너무 오래 걸린다면 이 값을 48로 변경한다
BATCH_SIZE = 16
STEPS_PER_EPOCH = 200
EPOCHS = 3

vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(INPUT_SIZE,INPUT_SIZE,3))

# 사전 학습된 레이어 고정
for layer in vgg16.layers:
    layer.trainable = False

# 신경망 맨 오른쪽에 노드 1개짜리 완전 연결 레이어 추가
input_ = vgg16.input
output_ = vgg16(input_)
last_layer = Flatten(name='flatten')(output_)
last_layer = Dense(1, activation='sigmoid')(last_layer)
model = Model(input=input_, output=last_layer)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

training_data_generator = ImageDataGenerator(rescale = 1./255)
testing_data_generator = ImageDataGenerator(rescale = 1./255)

training_set = training_data_generator.flow_from_directory(src+'Train/',
                                                target_size = (INPUT_SIZE, INPUT_SIZE),
                                                batch_size = BATCH_SIZE,
                                                class_mode = 'binary')

test_set = testing_data_generator.flow_from_directory(src+'Test/',
                                             target_size = (INPUT_SIZE, INPUT_SIZE),
                                             batch_size = BATCH_SIZE,
                                             class_mode = 'binary')

print("""
      Caution: VGG16 model training can take up to an hour if you are not running Keras on a GPU.
      If the code takes too long to run on your computer, you may reduce the INPUT_SIZE paramater in the code to speed up model training.
      """)

model.fit_generator(training_set, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose=1)

score = model.evaluate_generator(test_set, steps=100)

for idx, metric in enumerate(model.metrics_names):
    print("{}: {}".format(metric, score[idx]))
