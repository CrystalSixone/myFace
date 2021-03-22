from tools import *
import time

groupId = 'manager'
Users = {
    "wly":"王柳懿",
    "zx":"周逊",
    "wnj":"王乃佳"
}
# 请在faceRegest.py中注册后，在这里修改id和对应的中文姓名。

mode = 'local' # or 'baidu'
# mode == 'baidu' : cv2检测人脸 + 百度搜索这是谁,需要联网
# mode == 'local' : cv2注册人脸+检测,不用联网

model_name = 'model'

def detect_BD(frame,img):
    # 用百度API进行人脸检测（无搜索）
    faces = faceDetect(img)
    for face in faces:
        face.draw_bbox(frame)
    return faces

def detect_CV2(img):
    # 用cv2实现人脸检测
    faces = cv2Face(img)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+w),(255,0,0))
    return faces     

def recognition_BD(frame,img):
    # 用百度API进行人脸搜索（返回id）
    result = faceSearch(img,groupId)
    if result['result']:
        if result['Id'] in Users:
            Id = Users[result['Id']]
            return Id
        else:
            print('Id不在Users列表里，请检查')
            return None
    else:
        return None

if __name__ == '__main__':
    if mode == 'local':
        cap = cv2.VideoCapture(0)
        while(1):
            ret, frame = cap.read()
            cv2FaceTarget(frame,model_name,Users) # 训练好的模型名称
            cv2.imshow('face_detection',frame)
            if cv2.waitKey(1) & 0xff==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    else:
        cap = cv2.VideoCapture(0)
        current_time = time.time()
        while(1):
            ret,frame = cap.read()
            if ret:
                faces = detect_CV2(frame)
                if len(faces) >= 1: # 检测到有人
                    if time.time() - current_time >= 1: # 1s向百度发送一次
                        img = cv2_to_base64(frame)
                        user = recognition_BD(frame,img)
                        if user == None:
                            print('未检测到有效人脸')
                        else:
                            print('你好，{}!'.format(user))
                        current_time = time.time()

                cv2.imshow('test',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()