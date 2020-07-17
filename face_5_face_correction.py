import dlib
import cv2
import numpy as np
import os
import random
import glob
import math

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('models\shape_predictor_68_face_landmarks.dat')


# 获得人脸区域
def get_face_rect(rects):
    x = rects.left()
    y = rects.top()
    w = rects.right()
    h = rects.bottom()
    return x,y,w,h

# 人脸矫正
def correct_face(image,rects):
    shape=predictor(image,rects[0])
    x,y,w,h=get_face_rect(rects[0])
    # 获得左右眼的坐标
    x1,y1= shape.part(36).x, shape.part(36).y
    x2,y2 = shape.part(45).x, shape.part(45).y

    # 获取人脸区域
    face=image[y:h,x:w]
    width, height = face.shape[1], face.shape[0]

    # 获取左右眼的夹角
    h1=y2-y1
    w1=x2-x1
    a1=np.arctan(h1/w1)

    a=math.degrees(a1) #弧度转角度
    # print('旋转角度：%s°'%a)

    # 图像旋转，这里使用的角度制
    matRotate = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), a,1)
    dst = cv2.warpAffine(face, matRotate, (width, height))

    return dst

#得到人脸
def get_face(image_path,save=False):
    if type(image_path) is str:
        image = cv2.imread(image_path)
    else:
        image = image_path
        save = False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)  # 获得人脸个数
    # print(dets)
    face=None
    if len(dets)==0:
        pass
    else:
        face=correct_face(image, dets)
        if save:
            path=image_path.split('.')[0]
            cv2.imwrite(path+'.jpg',face)
    return face

# if __name__ == '__main__':
#     image_path='face_image\huge1.jpg'
#     face=get_face(image_path)
#     cv2.imshow('img',face)
#     cv2.waitKey(0)


