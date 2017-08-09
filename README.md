# LightNet: A Versatile, Standalone Matlab-based Environment for Deep Learning

Chengxi Ye, Chen Zhao, Yezhou Yang, Cornelia Fermüller, and Yiannis Aloimonos. 2016. LightNet: A Versatile, Standalone Matlab-based Environment for Deep Learning. In Proceedings of the 2016 ACM on Multimedia Conference (MM '16). Amsterdam, The Netherlands, 1156-1159. (http://dl.acm.org/citation.cfm?id=2973791)

![LightNet Icon](LightNet.png)

LightNet is a lightweight, versatile and purely Matlab-based deep learning framework. The aim of the design is to provide an easy-to-understand, easy-to-use and efficient computational platform for deep learning research. The implemented framework supports major deep learning architectures such as the Multilayer Perceptron Networks (MLP), Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). LightNet supports both CPU and GPU for computation and the switch between them is straightforward. Different applications in computer vision, natural language processing and robotics are demonstrated as experiments.

## How to use LightNet

Read the tutorial slides in the Documentations folder.  
Install the latest Matlab (**R2016b or later**) on a computer, and run the RunAll.m Matlab script.  
Have fun!  

## Recent updates

20170801: Second-order SGD is introduced (SGD2), together with the corresponding normalization technique RMSnorm and a new ModU activation function. 
SGD2 is a fast second-order training method (or known as the Newton's method) that trains faster and better, and shows better tolerance to bad initializations. 
Check it out in \SGD2. 

20-epoch training of a 10-layer-deep network. The network is initialized using Gaussian distribution with std from 10^(-4) to 10^4.

![Init](Init.png)

Implicit expansion is adopted to replace the bsxfun in LightNet. As a result, *Matlab R2016b* or later is required.

CUDNN is supported by installing Neural Network Toolbox from Mathworks. The convolutional network training is over 10x faster than the previous release! The current version can process 10,000 CIFAR-10 images per second in the training.

LightNet supports using pretrained ImageNet network models. 
![coco](coco.png)

Check CNN/Main_CNN_ImageNet_Minimal()

An example recognition using imagenet-vgg-f pretrained net:

![ImageNet Icon](ImageNetPreTrain.png)


## Major functions in LightNet
  
####network related:  
Main_Template: a template script used to train CNN and MLP networks.  
net_bp: implementation of the back propagation process which is used in CNN and MLP networks.  
net_ff: implementation of the feed forward process which is used in CNN and MLP networks.  
test_net: running the network in the testing mode to evaluate the current parameters.  
train_net: running the network in the training mode to evaluate and calculate the loss and gradients.  
TrainingScript: a training template for CNN and MLP networks.  
net_init*: how to initialize a neural network.  
  
####layers:  
bnorm: implementation of the batch normalization layer.  
conv_layer_1d: implementation of the 1d convolution layer. (CUDNN enabled)  
conv_layer_2d: implementation of the 2d convolution layer. (CUDNN enabled)  
dropout: implementation of the dropout layer.  
linear_layer: implementation of (fully-connected) linear layer. (CUDNN enabled)   
lrn: implementation of the local response normalization layer. (CUDNN enabled)  
maxpool: implementation of the 2d max-pooling layer. (CUDNN enabled)  
maxpool_1d: implementation of the 1d max-pooling layer. (CUDNN enabled)  
rmsnorm: implementation of the RMS normalization function.  
softmax: implementation of the softmax layer.  

####activation functions:  
leaky_relu: implementation of the leaky ReLU layer.
modu: implementation of the modulus unit layer.  
relu: implementation of the rectified linear unit layer.  
sigmoid_ln: implementation of the sigmoid layer.  
tanh_ln: implementation of the tanh layer.  

  
####loss functions:  
softmaxlogloss: implementation of the softmax log loss layer .  
  
####optimization related:  
adagrad: implementation of the Adagrad algorithm.  
adam: implementation of the Adam algorithm.  
rmsprop: implementation of the RMSProp algorithm.  
select_learning_rate: implementation of the Selective-SGD algorithm that automatically selects the optimal learning rate at the beginning or in the middle of the training.  
sgd: implementation of the stochastic gradient descent algorithm with momentum.  
sgd2: implementation of the second-order stochastic gradient descent algorithm with momentum.
  
####utility functions:  
generate_output_filename: generate output filename based on the current parameter settings.  
im2col_ln: customized im2col function used in the pooling layer.  
sig2col_ln: unroll overlapping signal windows into a matrix, used in the 1d pooling layer.  
pad_data*: a padding layer which is used in CNN.  
SwitchProcessor: a switch function between CPU and GPU.  


## How to accelerate LightNet  

Nvidia CUDNN can be used to calculate convolutions. 

1. You will need to install the Neural Network Toolbox from Mathworks. Make sure you can run it properly. (Ref to our tutorial slides.)  
2. Set opts.use_nntoolbox=1 in the main tesing script.  
