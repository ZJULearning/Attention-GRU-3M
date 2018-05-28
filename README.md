# Attention-GRU-3M
Code for the paper 'A Brand-level Ranking System with the Adapted Attention-GRU Model[C]'. (IJCAI 2018 Accepted)

Run command：python train.py --buckets "./data/" --checkpointDir ./log/ --exp debug --m1 1 --m2 0 --m3 1  
parameters：'buckets' is the folder for the input data，'checkpointDir' is the fold for the output data，'debug' is only a folder name（not important），'m1 m2 m3' indicate which Modification is activated.  
  
After running the above command for a while，run under the root direcotory：tensorboard --logdir="./log/"，then you can see the performance with a website link.  
  
virtual environment：virtualenv -p python2 env  
source env/bin/activate  
  
Install TensorFlow in Linux：  
pip install --upgrade  https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp27-none-linux_x86_64.whl  
  
Install TensorFlow in Mac：  
pip install --upgrade  https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py2-none-any.whl  
