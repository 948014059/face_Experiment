import cv2


face_date=cv2.CascadeClassifier('models\haarcascade_frontalface_default.xml')
def opencv_(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_date.detectMultiScale(gray,1.3,5)
    for face in faces:
        x,y,w,h=face
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    return img

if __name__ == '__main__':
    path='face_image\huge2.jpg'
    img=cv2.imread(path)
    img=opencv_(img)
    cv2.imshow('s',img)
    cv2.waitKey(0)