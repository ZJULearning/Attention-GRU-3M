# Attention-GRU-3M
  
执行语句是：python train.py --buckets "./data/" --checkpointDir ./log/ --exp debug --m1 1 --m2 0 --m3 1  
参数：buckets是输入数据文件目录，checkpointDir是输出数据文件目录，debug只是一个命名（不重要），m1 m2 m3指定激活哪些Modification。  
  
运行一段时间后，根目录下执行：tensorboard --logdir="./log/"，就可以通过网页查看模型运行效果了。  
  
虚拟环境：virtualenv env  
source env/bin/activate  
  
linux上安装TensorFlow：  
pip install --upgrade  https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp27-none-linux_x86_64.whl  
  
mac上安装TensorFlow：  
pip install --upgrade  https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py2-none-any.whl  
