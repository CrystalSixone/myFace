#### 1. 使用cv2本地检测+百度人脸搜索（需要联网）

1. **百度人脸注册**

   使用`faceRegist.py`中的`faceRegist_BD(img_path)`函数进行注册，传入参数：图像路径，用户名id。请注意：传入图像大小不能超过2M。

   注册完成后切换到`main.py`中，修改顶部Users内的信息。

2. **检测+搜索**

   修改`main.py`中的代码，`mode=='baidu'`

   运行`main`文件

#### 2. 使用cv2本地检测+人脸搜索(不需要联网)
1. **人脸采集**
	- 使用`faceRegist.py`中的`faceRegist_local(user_id,model_name)`函数进行人脸采集+训练;
	- 其中,`cv2faceSample(user_id)`可用于人脸采集.如有要注册多人,请传入不同的user_id来重复调用该函数.采集到的人像自动保存在`/Facedata/xxx`中;
	- 采集完后切换到`main.py`中，修改顶部Users内的信息。
	
2. **人脸训练**

   `cv2faceTrain(model_name)`,传入模型名进行训练.训练完成的文件自动保存为`xxx.xml`.

   同时也要修改`main.py`中的`model_name`为刚才训练的模型名,否则无法正常加载模型.

3. **检测+搜索**

   修改`main.py`中的代码，`mode=='local'`

   运行`main`文件