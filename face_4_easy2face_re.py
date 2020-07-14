# 导入easyfaces
from face_3_easyfaces import EasyFaces
import cv2
# 实例化
easyface=EasyFaces()
# 打开摄像头
easyface.openCamera()
name=input('please input your name:')
# # 图像预处理，以及人脸检测
easyface.faceDetection(gray=False,name=name)
easyface.Video_recognition()

#图片地址
# face_path=r'D:\pythonFolder\Opencv_teach\opencv_\face_file\huge.jpg'
# # opencv读取图片
# image=cv2.imread(face_path)
# # 人脸识别
# name=easyface.Pic_recognition(face_path)
# # opencv在图片上添加文字
# cv2.putText(image,name,(10,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
# # 显示图片
# cv2.imshow('ss',image)
# cv2.waitKey(0)
# print(name)
# easyface.Video_recognition()