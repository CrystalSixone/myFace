# 利用百度进行人脸注册
# 每注册一个新人，请修改img_path和userId

from tools import *

def faceRegist_BD(img_path,userId):
    """使用百度api进行人脸注册，注册完成后可以使用百度api进行人脸搜索"""
    img_path = 'wnj.jpg'
    img = image_to_base64(img_path)
    groupId = 'manager'
    userId = userId
    faceRegister(img,groupId,userId)

def faceRegist_local(user_id,model_name):
    """使用opencv本地训练人脸库"""
    #采集人像，如有多人请先将每个人都单独采集好再使用cv2faceTrain进行训练
    cv2faceSample('wly') # 输入：当前采集人像的id.
    #训练
    cv2faceTrain('model') # 输入: 模型名称.


if __name__ == '__main__':
    user_id = 'wly'
    model_name = 'model'
    faceRegist_local(user_id,model_name)