import numpy as np
import cv2
import dlib

predictor_model = 'models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()

def get_image_hull_mask(image_shape, image_landmarks, ie_polys=None):
    print(image_shape,image_landmarks.shape)
    # get the mask of the image
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')

    int_lmrks = np.array(image_landmarks, dtype=np.int)

    #hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)
    # 填充一个全0 矩阵 与原图宽高相同
    hull_mask = np.full(image_shape[0:2] + (1,), 0, dtype=np.float32)

    # 往矩阵中填充68个特征点坐标的的值 为1
    cv2.fillConvexPoly(hull_mask,cv2.convexHull(int_lmrks[0:67]),(1,))
    return hull_mask

def get_landmarks(image):
    predictor = dlib.shape_predictor(predictor_model)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 0)
    rect=rects[0]

    x = rect.left()
    y = rect.top()
    w = rect.right()
    h = rect.bottom()

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(image, rects[i]).parts()])
    return (x,y,w,h),landmarks

def get_seg_face(img_path):
    image = cv2.imread(img_path)
    # 获得人脸矩形区域和68个特征点的坐标
    (x,y,w,h),landmarks = get_landmarks(image)

    mask = get_image_hull_mask(np.shape(image), landmarks).astype(np.uint8)
    image = cv2.add(image, np.zeros(np.shape(image), dtype=np.uint8),mask=mask)
    # 更换底色
    image[image[...,0]==0]=255.
    image=image[y:h,x:w]
    cv2.imshow('s2',image)
    cv2.waitKey(0)

if __name__ == "__main__":
    get_seg_face(r'face_image/huge2.jpg')