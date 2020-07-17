import dlib
import cv2
import numpy as np
import math

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

# 获得人脸区域
def get_face_rect(rects):
    x = rects.left()
    y = rects.top()
    w = rects.right()
    h = rects.bottom()
    return x,y,w,h

# 获得人脸旋转后的坐标
def get_trainpose_point(x,y,w,h,angle):
    # 求三角函数值 这里默认使用弧度制，所以输入的是弧度
    sina=math.sin(angle)
    cosa=math.cos(angle)

    # 获得矩形的宽高
    height=h-y
    weidth=w-x

    # 获得中心点坐标
    centerx=int(x+weidth/2)
    centery=int(y+height/2)

    # 分别获得当前 左上角 右上角 右下角的坐标
    left_point=np.array([x,y])
    top_right_point=np.array([w,y])
    bottom_right_point=np.array([w,h])

    # 组合
    points=np.concatenate((left_point,top_right_point,bottom_right_point))

    # 分别获得旋转后的左上角右上角 右下角的坐标
    points[0]=(points[0] - centerx) * cosa - (points[1] - centery) * sina + centerx
    points[1]=(points[0] - centerx) * sina + (points[1] - centery) * cosa + centery

    points[2] = (points[2] - centerx) * cosa - (points[3] - centery) * sina + centerx
    points[3] = (points[2] - centerx) * sina + (points[3] - centery) * cosa + centery

    points[-2]=(points[-2] - centerx) * cosa - (points[-1] - centery) * sina + centerx
    points[-1]=(points[-2] - centerx) * sina + (points[-1] - centery) * cosa + centery

    return points.reshape(-1,2)

# 人脸矫正
def correct_face(image,rects,size=128):
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

    a = math.degrees(a1)  # 弧度转角度
    # print('旋转角度：%s°' % a)

    # 这里使用弧度制
    points=get_trainpose_point(x,y,w,h,a1)
    points=np.array(points,np.float32)

    # 将 旋转后的坐标 仿射变换到新的坐标
    new_point=np.array([[0,0],[size,0],[size,size]],np.float32)
    A1=cv2.getAffineTransform(points,dst=new_point)
    d1=cv2.warpAffine(image,A1,(size,size),borderValue=125)

    return d1

#得到人脸
def get_face(image_path,save=False):
    if type(image_path) is str:
        image=cv2.imread(image_path)
    else:
        image=image_path
        save=False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)  # 获得人脸个数
    # print(dets)
    face=None
    if len(dets)==0:
        print('未检测到人脸')
    else:
        face=correct_face(image, dets)
        if save:
            path=image_path.split('.')[0]
            cv2.imwrite(path+'.jpg',face)
    return face

# if __name__ == '__main__':
#     image_path = r'face_image/huge2.jpg'
#     face = get_face(image_path)
#     cv2.imshow('img', face)
#     cv2.waitKey(0)