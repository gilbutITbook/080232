name = input("What is your name?")

import os
import sys
import cv2
import utils
from keras.models import load_model
import face_detection
import collections

# 훈련시킨 샴 신경망 모델이 있는지 검사한다
files = os.listdir()
if 'siamese_nn.h5' not in files:
    print("Error: Pre-trained Neural Network not found!")
    print("Please run siamese_nn.py first")
    sys.exit()

# 온보딩 과정을 거쳤는지 확인한다
if 'true_img.png' not in files:
    print("Error: True image not found!")
    print("Please run onbarding.py first")
    sys.exit()

# 미리 훈련시킨 샴 신경망을 가져온다
model = load_model('siamese_nn.h5', custom_objects={'contrastive_loss': utils.contrastive_loss, 'euclidean_distance': utils.euclidean_distance})

# 온보딩 과정으로 얻은 정답 이미지를 준비한다
true_img = cv2.imread('true_img.png', 0)
true_img = true_img.astype('float32')/255
true_img = cv2.resize(true_img, (92, 112))
true_img = true_img.reshape(1, true_img.shape[0], true_img.shape[1], 1)

video_capture = cv2.VideoCapture(0)
preds = collections.deque(maxlen=15)

while True:
    # 프레임별로 이미지를 찍는다
    _, frame = video_capture.read()

    # 얼굴을 검출한다
    frame, face_img, face_coords = face_detection.detect_faces(frame, draw_box=False)

    if face_img is not None:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img.astype('float32')/255
        face_img = cv2.resize(face_img, (92, 112))
        face_img = face_img.reshape(1, face_img.shape[0], face_img.shape[1], 1)
        preds.append(1-model.predict([true_img, face_img])[0][0])
        x,y,w,h = face_coords
        if len(preds) == 15 and sum(preds)/15 >= 0.3:
            text = "Identity: {}".format(name)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
        elif len(preds) < 15:
            text = "Identifying ..."
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 5)
        else:
            text = "Identity Unknown!"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
        frame = utils.write_on_frame(frame, text, face_coords[0], face_coords[1]-10)

    else:
        preds = collections.deque(maxlen=15) # 얼굴이 없다면 결과를 지운다

    # 결과 프레임을 보여준다
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 다 마치면 찍은 프레임을 릴리즈한다
video_capture.release()
cv2.destroyAllWindows()
