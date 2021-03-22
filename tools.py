# -*- coding: utf-8 -*-
import os,sys 
import requests
import base64
import cv2
import json
import numpy as np
from aip import AipFace

# 61的百度控制台
APP_ID = '23814423'
API_KEY = '4CemeBj81lwefhYx8vs1Wdgm'
SECRET_KEY = 'w6GLrq91qjsL711QaYKHqfvReLl1n68O'
client = AipFace(APP_ID, API_KEY, SECRET_KEY)

# cv2实现人脸检测
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# model = cv2.face.EigenFaceRecognizer_create() # 用于训练的模型
model = cv2.face.LBPHFaceRecognizer_create() # 用于训练的模型
imgPath = 'Facedata' # 采集图像的默认保存路径

class Face(): # 用于存储脸部信息
    def __init__(self,face_token,location,age,beauty,expression,gender):
        self.face_token = face_token
        self.location = location # left,top,width,height,rotation
        self.age = age
        self.beauty = beauty # 0-100，数值越大越美
        self.expression = expression # none:不笑；smile:微笑；laugh:大笑
        self.gender = gender # male:男性 female:女性
    
    def draw_bbox(self,img): # 画出检测框
        x,y,w,h = int(self.location['left']),int(self.location['top']),int(self.location['width']),int(self.location['height'])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    def draw_userid(self,img,userid): # 打出匹配的名字
        x,y = int(self.location['left']),int(self.location['top'])
        cv2.putText(img,userid,(x-20,y-20),cv2.FONT_HERSHEY_COMPLEX,5,(0,255,255),2)  

def image_to_base64(image_path):
    '''
    :param image_path: the path of the image
    :return base64
    '''
    # 将本地图片转换为base64编码
    with open(image_path,'rb') as f:
        img = f.read()
        return str(base64.b64encode(img),encoding='utf-8')

def cv2_to_base64(img):
    '''
    :param img: the image read by cv2
    :return base64
    '''
    # 将通过cv2读取的图片转化为base64格式
    image = cv2.imencode('.jpg',img)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code

def faceRegister(img,groupId,userId):
    """在百度AI中注册人脸库"""
    # img = str(base64.b64encode(img), "utf-8")
    imageType="BASE64"
    groupId = groupId
    userId = userId
    result = client.addUser(img,imageType,groupId,userId)
    print(result)
    if result["error_msg"] == "SUCCESS":
        print("注册成功！")
        return True
    else:
        print("注册失败，{}".format(result))
        return None

def faceDetect(img):
    """人脸检测，最多检测3人"""
    # img = base64.b64decode(img.encode('utf-8'))
    # print(img[0:20])
    """ 如果有可选参数 """
    options = {}
    options["face_field"] = "age,beauty,expression,gender"
    options["max_face_num"] = 3
    options["face_type"] = "LIVE"
    options["liveness_control"] = "LOW"

    result = client.detect(img,"BASE64",options)
    face_num = result['result']['face_num']
    print('检测到{}人'.format(face_num))
    faces = []
    for face in result['result']['face_list']:
        face_token = face['face_token']
        location = face['location']
        age = face['age']
        gender = face['gender']['type']
        beauty = face['beauty']
        expression = face['expression']['type']
        new_face = Face(face_token=face_token,location=location,age=age,beauty=beauty,expression=expression,gender=gender,emotion=emotion,mask=mask)
        faces.append(new_face)
    for i,face in enumerate(faces):
        print('第{}人\t年龄：{}岁\t性别：{}\t'.format(str(i),face.age,face.gender))
    return faces

def faceSearch(img, groupId):
    """在人脸库中进行匹配,最多搜索1人"""
    # image = str(base64.b64encode(img), "utf-8")
    result = client.search(img, "BASE64", groupId)
    if result["error_msg"] == "SUCCESS":
        Id = result["result"]["user_list"][0]["user_id"]		# 人脸id
        group = result["result"]["user_list"][0]["group_id"]    # 所属组别
        score = result["result"]["user_list"][0]["score"]		# 得分
        # location = result["result"]["face_list"][0]["location"] # 人脸位置
        return {'result':True,'Id':Id}
    if result["error_msg"] == 'liveness check fail':
        print("未通过活体检测")
        return {'result':False}
    else:
        print("未检测到人脸")
        return {'result':False}

def cv2Face(img):
    """用cv2直接实现人脸检测"""
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.3,5)
    return faces

def cv2FaceTarget(img,model_name,Users=''):
    """用cv2实现带目标的人脸识别，请先用cv2faceSample和cv2faceTrain进行人脸采集和训练"""
    names = []
    for subdir in os.listdir(imgPath):
        subpath=os.path.join(imgPath,subdir)
        if os.path.isdir(subpath):
            names.append(subdir)
    try:
        model.read('{}.xml'.format(model_name))
    except Exception as e:
        print('模型加载失败!请检查传入的模型名称是否正确')
        return 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.3,5)
    if len(faces)>=1:
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0))
            face_image = gray[y:y+h,x:x+w]
            face_image = cv2.resize(face_image,(92,112),interpolation = cv2.INTER_CUBIC)

            idnum,confidence = model.predict(face_image)
            if confidence > 50:
                if names[idnum] in Users:
                    name = names[idnum]
                    print("你好，{}!".format(Users[name]))
                else:
                    print("当前人脸不在人脸库定义中,请检查")
            else:
                name = "Unknown"
            # confidence = "{0}%".format(round(100 - confidence))
            confidence = "{}%".format(round(confidence))
            cv2.putText(img, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, str(confidence), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

def cv2faceSample(face_id):
    save_frames = 0
    max_frame_num = 100 # 每人采集100张照片
    cap = cv2.VideoCapture(0)
    path=os.path.join(imgPath,str(face_id))
    #若没有路径，则建立路径
    if os.path.isdir(path) == 0:
        os.mkdir(path)
    while True:
        ret, img = cap.read()
        if ret:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray,1.3,5)
            if len(faces) >= 1:
                for(x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+w),(255,0,0))     
                    face_image = gray[y:y+h,x:x+w]
                    face_image = cv2.resize(face_image,(92,112),interpolation = cv2.INTER_CUBIC)
                    cv2.imwrite('%s/%s.png'%(path,str(save_frames)),face_image)
                save_frames += 1
                if save_frames > max_frame_num:
                    break
                print("save_frames"+str(save_frames))
            cv2.imshow('faceSample',img)
            if cv2.waitKey(10) & 0xff==ord('q'):
                cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    print('采集完毕！人脸图片已保存至{},此目标的id为{}'.format(path,str(face_id)))

def cv2faceTrain(model_name):
    images = []
    labels = []
    label = 0
    for subdir in os.listdir(imgPath):
        subpath=os.path.join(imgPath,subdir)
        if os.path.isdir(subpath):
            for filename in os.listdir(subpath):
                imgpath=os.path.join(subpath,filename)
                img=cv2.imread(imgpath,cv2.IMREAD_COLOR)
                gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                images.append(gray_img)
                labels.append(label)
            print('label:{},id:{}'.format(label,subdir))
            label+=1
            
    images=np.asarray(images)
    labels=np.asarray(labels)
    model.train(images,labels)
    model.save("{}.xml".format(model_name))
    print('训练完成！模型已保存在{}.xml'.format(model_name))

if __name__ == '__main__':
    # cv2faceSample('wly')
    # cv2faceTrain('wly')

    cap = cv2.VideoCapture(0)
    while(1):
        ret, frame = cap.read()
        cv2FaceTarget(frame,'wly')
        cv2.imshow('face_detection',frame)
        if cv2.waitKey(1) & 0xff==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # img_path = 'wly.jpg'
    # img = cv2.imread(img_path)
    # cv2FaceTarget(img,'wly')
    # cv2.imshow('test',img)
    # if cv2.waitKey(0) & 0xff==ord('q'):
    #     cv2.destroyAllWindows()
    