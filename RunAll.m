clear all;
cd('./MLP');
disp('Testing training a multilayer perceptron.')
Main_MNIST_MLP_RMSPROP();
cd ..


clear all;

cd('./ReinforcementLearning');
disp('Testing training a Q-network.')
Main_Cart_Pole_Q_Network
cd ..


clear all;
cd('./RNN');
disp('Testing training an LSTM.')
Main_Char_RNN();
cd ..


clear all;
cd('./CNN');
disp('Testing using a pretrained ImageNet convolutional neural network model.')
Main_CNN_ImageNet_minimal();
cd ..

clear all;
cd('./CNN');
disp('Testing training a new convolutional neural network.')
disp('An Nvidia GPU is required by default.')
disp('If you do not have it please set use_gpu=0')
disp('Neural Network Toolbox is required by default.')
disp('If you do not have it please set opts.use_nntoolbox=0')
Main_CIFAR_CNN_SGD();
cd ..
