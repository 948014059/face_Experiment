import numpy as np
import tensorflow.keras as k
import cv2
from face_6_face_transformation import get_face


# l2 正则化
def l2_normalize(x,axis=1,epsilon=1e-10):
    output=x/np.sqrt(np.maximum(np.sum(np.square(x),axis=axis,keepdims=True),epsilon))
    return output


# 距离计算
def person_distance(person_encodings,person_unknow):
    if len(person_encodings)==0:
        return np.empty((0))
    # distances = np.sqrt(np.sum(np.asarray(person_unknow - person_encodings) ** 2, axis=1))
    return np.linalg.norm(person_encodings-person_unknow,axis=1)


# 判断距离是否小于阈值 
def com_person(person_list,person,tolerance=0.9):
    dis=person_distance(person_list,person)
    # print(dis)
    return list(dis <= tolerance)

def face_embeddings(models,img):
    face=get_face(img)
    face=cv2.resize(face,(160,160))
    face=face.reshape(-1,160,160,3)/255.
    embeding=models.predict(face)
    embeding=l2_normalize(embeding)
    # print(embeding)
    return embeding


# if __name__ == '__main__':
#     model=k.models.load_model('models/facenet_keras.h5')
#     img_path1=r'E:\DataSets\Face_net\all_faces\006_3.jpg'
#     img_path2=r'E:\DataSets\Face_net\all_faces\006_2.jpg'
#     img_path3=r'E:\DataSets\Face_net\all_faces\050_3.jpg'
#
#     img1=cv2.imread(img_path1)
#     img2=cv2.imread(img_path2)
#     img3=cv2.imread(img_path3)
#
#     emb1=face_embeddings(model,img1).reshape([128])
#     emb2=face_embeddings(model,img2).reshape([128])
#     emb3=face_embeddings(model,img3).reshape([128])
#
#     s=person_distance([emb1,emb3],emb2)
#     print(s)