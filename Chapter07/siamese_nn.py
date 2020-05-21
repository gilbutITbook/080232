'''
안면 인식에 사용할 샴 신경망을 훈련시키는 메인 코드
'''
import utils
import numpy as np
from keras.layers import Input, Lambda
from keras.models import Model

faces_dir = 'att_faces/'

# 훈련 데이터셋과 테스트 데이터셋을 가져온다
(X_train, Y_train), (X_test, Y_test) = utils.get_data(faces_dir)
num_classes = len(np.unique(Y_train))

# 샴 신경망을 만든다
input_shape = X_train.shape[1:]
shared_network = utils.create_shared_network(input_shape)
input_top = Input(shape=input_shape)
input_bottom = Input(shape=input_shape)
output_top = shared_network(input_top)
output_bottom = shared_network(input_bottom)
distance = Lambda(utils.euclidean_distance, output_shape=(1,))([output_top, output_bottom])
model = Model(inputs=[input_top, input_bottom], outputs=distance)

# 모델을 훈련시킨다
training_pairs, training_labels = utils.create_pairs(X_train, Y_train, num_classes=num_classes)
model.compile(loss=utils.contrastive_loss, optimizer='adam', metrics=[utils.accuracy])
model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_labels,
          batch_size=128,
          epochs=10)

# 모델을 저장한다
model.save('siamese_nn.h5')
