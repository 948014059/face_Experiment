from  face_9_distance import face_embeddings,person_distance,com_person
import tensorflow.keras as k
import cv2
import os
from PIL import  Image
import numpy as np
def search_face(mode,face,face_dir):
    this_face_emb=face_embeddings(model,face)
    image_paths=[os.path.join(face_dir,p)for p in os.listdir(face_dir)]
    face_img=[]
    for index,image_path in enumerate(image_paths):
        image=cv2.imread(image_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        emb=face_embeddings(mode,image)
        s = person_distance(this_face_emb,emb)
        if s[0] <0.85:
            image=cv2.resize(image,(256,256))
            face_img.append(image)
        if index>50:
            break
        print('当前检测图片：%s,人脸距离：%s'%(image_path,s[0]))

    return face_img

if __name__ == '__main__':
    model = k.models.load_model('models/facenet_keras.h5')
    img_path = r'E:\DataSets\Face_net\all_faces\006_3.jpg'
    face=cv2.imread(img_path)
    img_lists=search_face(model,face,r'E:\DataSets\Face_net\all_faces')

    img_len=len(img_lists)
    new_img=Image.new('RGB',(256*3,(img_len//3)*256+256),(0,0,0))
    print(np.array(new_img).shape,img_len)

    for index,img in enumerate(img_lists):
        img=Image.fromarray(img)
        x=index % 3
        y=int(index/3)
        new_img.paste(img,(x*256,y*256))
    new_img.show()
