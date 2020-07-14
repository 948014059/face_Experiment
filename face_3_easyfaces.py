import face_recognition
import cv2
import os
import dlib
class EasyFaces():
    def __init__(self):
        self.image=None
        self.camera=None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.name=[]
        self.encodings=[]
    def openCamera(self,camnum=0):
        '''
        try to open camera ,camnum is the camera index,
        if you not have camera ,it will be open failure,
        if you have more camera, please try more index.
        :param camnum:
        :return:
        '''
        try:
            self.camera=cv2.VideoCapture(camnum)
        except:
            print('Failed to open the camera, please check if there is a connected camera')
    def faceDetection(self,gray=False,close='q',save='x',name=None):
        '''
        :param gray:it can converts images to grayscale
        :param close:when you press close ,camera will be release
        :param save: when you press save ,it will be save the face picture
        tips: Please keep a face when you press Save!
        :param name: This is the face name
        :return:
        '''
        while True:
            _, self.image = self.camera.read()
            if gray == True:
                image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                image=self.image
            dets = self.detector(image, 1)
            for k, d in enumerate(dets):
                # cv2.rectangle(img,(d.left(),d.bottom()),(d.right(),d.top()),(0,255,0),2)
                shape = self.predictor(image, d)
                for i in range(68):
                    cv2.circle(image,
                               (shape.part(i).x, shape.part(i).y),
                               4, (0, 255, 0), -1, 8)
            cv2.putText(image,'Please press %s to quit!'%close,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
            cv2.putText(image,'Please press %s to save!'%save,(10,40),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),1)
            cv2.imshow('camera', image)
            flag = cv2.waitKey(1)
            if flag == ord(close):
                break
            if flag == ord(save):
                if not os.path.exists('face_file'):
                    os.mkdir('face_file')
                cv2.imwrite('face_file/%s.jpg'%name,self.image)
        # self.camera.release()
        cv2.destroyAllWindows()

    def showVideo(self):
        '''show video'''
        while True:
            _, self.image = self.camera.read()
            image=self.image
            cv2.putText(image,'Please press q to quit!',(10,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
            cv2.imshow('camera', image)
            flag = cv2.waitKey(1)
            if flag == ord('q'):
                break
        self.camera.release()
        cv2.destroyAllWindows()

    def Pic_recognition(self,face_path):
        '''
        Picture comparison face recognition
        :param face_path:
        :return:
        '''
        if os.path.exists('face_file'):
            self.pic()
            unknown_picture = face_recognition.load_image_file(face_path)
            unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]
            results = face_recognition.compare_faces(self.encodings, unknown_face_encoding)
            if True in results:
                index=results.index(True)
                return self.name[index]
            else:
                return 'not found face'
        else:
            print('not found file')

    def Video_recognition(self):
        '''
        Turn on the camera for face recognition
        :return:
        '''
        if os.path.exists('face_file'):
            self.pic()
            process_this_frame = True
            while True:
                ret, img = self.camera.read()
                small_frame = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)  # 缩小图片 方便处理
                if process_this_frame:
                    face_locations = face_recognition.face_locations(small_frame)
                    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
                    # print(face_locations,'---',face_encodings)
                    face_names = []
                    for face_encoding in face_encodings:
                        match = face_recognition.compare_faces(self.encodings, face_encoding, tolerance=0.5)
                        if True in match:
                            index = match.index(True)
                            # print(index)
                            face_names.append(self.name[index])
                        # print(name)
                process_this_frame = not process_this_frame
                for (top, right, bottom, left), names in zip(face_locations, face_names):
                    cv2.rectangle(img, (left * 5, top * 5), (right * 5, bottom * 5), (0, 255, 0), 2)  # 放大图片
                    cv2.putText(img, names, (left * 5, top * 5), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 255), 1)
                cv2.imshow('video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.camera.release()
            cv2.destroyAllWindows()
        else:
            print('not found file')

    def pic(self):
        '''
        Return the image data in the folder
        :return:
        '''
        image_lists = [os.path.join(r'.\face_file', path) for path in os.listdir('face_file')]
        print(image_lists)
        self.name = []
        self.encodings = []
        for image_list in image_lists:
            if image_list.split('.')[-1] == 'jpg':
                self.name.append(image_list.split('.')[1].replace('\\', ' ').split(' ')[-1])
                load_img = face_recognition.load_image_file(image_list)
                self.encodings.append(face_recognition.face_encodings(load_img)[0])




