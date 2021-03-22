#!/usr/bin/env python
# -*- coding: utf-8 -*-
#author : 61
#说明：定义人脸检测的类及类函数


import os
import sys 
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
import cv2 
import numpy as np
import shutil
import time


class FaceInfo(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.move_threshold = 20
        self.middle_X = 320
        self.middle_Y = 180
    
    def get_faceinfo(self):
        return [self.x,self.y,self.w,self.h]
    
    def put_faceinfo(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class FaceDetector(object):
    def __init__(self):
        # flag
        self.face_id = "zhy"
        self.detect_mode = 0 # 0:preprocess & training & face_Detection #1:only face_Detection # 2:only face_detection without target &smile detection
        self.action_mode = 0
        self.find_people_flag = 0 #0:not find #1:find
        self.smile_count = 0
        self.interaction_validation = 0
        self.save_frames = 0
        self.write_flag = 0
        self.sendnav_flag = 1
        self.video_name = "output.avi"
        self.face_count = 0
        self.face_begin_time = 0
        self.fps = 0.0
        self.fps_total = []
        self.t1 = 0.0

        # opencv_face_detect:
        self.faceCascade = cv2.CascadeClassifier(r'/home/w61/ros/flight/src/show_image/data/haarcascade_frontalface_default.xml')
        self.smileCascade = cv2.CascadeClassifier(r'/home/w61/ros/flight/src/show_image/data/haarcascade_smile.xml')
        #self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model = cv2.face.EigenFaceRecognizer_create()
        self.imgPath = "/home/w61/ros/flight/src/show_image/src/Facedata"

        #opencv_image:
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.crop_size = (92,112)
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter(self.video_name,self.fourcc,20.0,(640,360))
        
        #ros:
        self.bridge = CvBridge()
        self.cmdvel_pub = rospy.Publisher('cmd_vel',Twist,queue_size = 1)
    
    def create_out(self):
        self.out = cv2.VideoWriter(self.video_name,self.fourcc,20.0,(640,360))
        #视频输出名称默认为“output.avi”

    def img2cv2(self,data):
        img = self.bridge.imgmsg_to_cv2(data,"bgr8")
        return img

    def face_training(self):
        images = []
        labels = []
        label = 0

        for subdir in os.listdir(self.imgPath):
            subpath=os.path.join(self.imgPath,subdir)
            if os.path.isdir(subpath):
                for filename in os.listdir(subpath):
                    self.imgpath=os.path.join(subpath,filename)
                    img=cv2.imread(self.imgpath,cv2.IMREAD_COLOR)
                    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    images.append(gray_img)
                    labels.append(label)
                label+=1
        images=np.asarray(images)
        labels=np.asarray(labels)

        self.model.train(images,labels)
        self.model.save(r'/home/w61/ros/flight/src/show_image/src/XML/'+self.face_id+".xml")
        self.save_frames += 1
        
    def img_preprocess(self,img):
        path=os.path.join(self.imgPath,str(self.face_id))
        #若没有路径，则建立路径
        if os.path.isdir(path) == 0:
            os.mkdir(path)
        
        if self.save_frames <= 25:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray,1.3,5)
            for(x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+w),(255,0,0))     
                face_image = gray[y:y+h,x:x+w]
                face_image = cv2.resize(face_image,self.crop_size,interpolation = cv2.INTER_CUBIC)
                cv2.imwrite('%s/%s.png'%(path,str(self.save_frames)),face_image)
                self.save_frames += 1
                print("save_frames"+str(self.save_frames))
            cv2.imshow('preprocess',img)
            if cv2.waitKey(10) & 0xff==ord('q'):
                cv2.destroyAllWindows()
            
            if self.write_flag == 1:
                self.out.write(img)
    
    def face_detect_simplest(self,img):
        face_infor = FaceInfo()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray,1.3,5)
        if len(faces) > 0:
            self.find_people_flag = 1
        else:
            self.find_people_flag = 0
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+w),(255,0,0))
            face_infor.put_faceinfo(x,y,w,h)
        cv2.imshow('face_detection',img)

        if cv2.waitKey(10) & 0xff==ord('q'):
            cv2.destroyAllWindows()

        if self.write_flag == 1:
            self.out.write(img)
        
        return face_infor

    def face_detect_with_target(self,img):
        self.t1 = time.time()
        face_infor = FaceInfo()
        names = []
        for subdir in os.listdir(self.imgPath):
            subpath=os.path.join(self.imgPath,subdir)
            if os.path.isdir(subpath):
                names.append(subdir)

        self.model.read(r'/home/w61/ros/flight/src/show_image/src/XML/'+self.face_id+'.xml')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray,1.3,5)
        if len(faces) > 0:
            self.find_people_flag = 1
        else:
            self.find_people_flag = 0
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+w),(255,0,0))
            face_image = gray[y:y+h,x:x+w]
            face_image = cv2.resize(face_image,self.crop_size,interpolation = cv2.INTER_CUBIC)
    
            idnum,confidence = self.model.predict(face_image)
            if confidence < 90:
                name = names[idnum]
                print("I see " + str(name) + "...")
            else:
                name = "Unknown"
            confidence = "{0}%".format(round(100 - confidence))
            cv2.putText(img, name, (x+5, y-5), self.font, 1, (0, 0, 255), 2)
            cv2.putText(img, str(confidence), (x+5, y+h-5), self.font, 1, (0, 0, 0), 2)
        
            if name == self.face_id:
                face_infor.put_faceinfo(x,y,w,h)

        self.fps = ( self.fps + (1./(time.time()-self.t1)) ) / 2
        if self.interaction_validation == 0:
            self.fps_total.append(self.fps)
        cv2.putText(img, "fps:"+str(self.fps), (20, 20), self.font, 1, (0, 0, 255), 2)

        cv2.imshow('face_detection',img)
        if cv2.waitKey(10) & 0xff==ord('q'):
            cv2.destroyAllWindows()

        if self.write_flag == 1:
            self.out.write(img)
        
        return face_infor

    def face_detect_with_smile(self,img):
        self.t1 = time.time()
        face_infor = FaceInfo()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray,1.3,5)
        if len(faces) > 0:
            self.find_people_flag = 1
        else:
            self.find_people_flag = 0
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+w),(255,0,0))
            face_infor.put_faceinfo(x,y,w,h)
            # 微笑检测
            if self.interaction_validation == 0:
                gray_image = gray[y:y+h,x:x+w]
                color_image = img[y:y+h,x:x+w]
                smiles = self.smileCascade.detectMultiScale(gray_image,scaleFactor= 1.16,minNeighbors=65,flags=cv2.CASCADE_SCALE_IMAGE)
                for(sx,sy,sw,sh) in smiles:
                    cv2.rectangle(color_image,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
            
                if self.sendnav_flag == 1:
                    if len(smiles) > 0:
                        if self.smile_count < 5: #检测到5次微笑才算检测成功
                            self.smile_count += 1
                        else:
                            self.interaction_validation = 1
                            fs = 0.0
                            for f in self.fps_total:
                                fs = fs + f
                            fs = fs / len(self.fps_total)
                            print("fps_final:"+str(fs))
                    else:
                        self.interaction_validation = 0
                self.send_isnav(self.interaction_validation)

        self.fps = ( self.fps + (1./(time.time()-self.t1)) ) / 2
        if self.interaction_validation == 0:
            self.fps_total.append(self.fps)
        cv2.putText(img, "fps:"+str(self.fps), (20, 20), self.font, 1, (0, 0, 255), 2)
        cv2.imshow('face_detection',img)
        
        if cv2.waitKey(10) & 0xff==ord('q'):
            cv2.destroyAllWindows()

        if self.write_flag == 1:
            self.out.write(img)
        
        return face_infor

