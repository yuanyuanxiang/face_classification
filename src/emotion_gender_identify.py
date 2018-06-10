"""
对图像中的人脸进行检测，然后识别其性别、表情。
用法：python image_emotion_gender_demo.py IMAGE_FILE
源代码摘自：face_classification\src
2018年6月10日 by yuanyuanxiang
"""

import sys
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

# parameters for loading data and images
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}
gender_labels = {0:'woman', 1:'man'}

# hyper-parameters for bounding boxes shape
gender_offsets = (10, 10)
emotion_offsets = (0, 0)
face_coordinates = [24, 24, 112, 112]
x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
s1, s2, t1, t2 = apply_offsets(face_coordinates, emotion_offsets)

# loading models
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# 检测图像数据
def test_src(image_src):
    rgb_image = np.array(image_src).astype(np.float32)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')

    rgb_face = rgb_image[y1:y2, x1:x2]
    gray_face = gray_image[t1:t2, s1:s2]

    try:
        rgb_face = cv2.resize(rgb_face, (gender_target_size))
        gray_face = cv2.resize(gray_face, (emotion_target_size))
    except:
        return ''

    rgb_face = preprocess_input(rgb_face, False)
    rgb_face = np.expand_dims(rgb_face, 0)
    gender_prediction = gender_classifier.predict(rgb_face)
    gender_label_arg = np.argmax(gender_prediction)
    gender_text = gender_labels[gender_label_arg]

    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion_text = emotion_labels[emotion_label_arg]
    result = emotion_text + '_' + gender_text
    return result

# 检测图像文件
def test_image(image_file):
    try:
        image = Image.open(image_file)
        print('>>> Run test on image:', image_file)
    except IOError:
        print('IOError: File is not accessible.')
        return
    result = test_src(image)
    print('Result =', result)

# MAIN
if __name__ == '__main__':
    test_image('image.jpg' if (1 == len(sys.argv)) else sys.argv[1])
