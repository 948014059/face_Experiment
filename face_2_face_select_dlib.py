import cv2
import dlib

predictor=dlib.shape_predictor('models\shape_predictor_68_face_landmarks.dat')
detector=dlib.get_frontal_face_detector()

def dlib(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dets=detector(gray,1)
    for k,v in enumerate(dets):
        shape=predictor(img,v)
        for i in range(68):
            cv2.circle(img,(shape.part(i).x,shape.part(i).y),1,(0,0,255),-1,8)
    return img


if __name__ == '__main__':
    path='face_image\huge2.jpg'
    img=cv2.imread(path)
    img=dlib(img)
    cv2.imshow('s',img)
    cv2.waitKey(0)