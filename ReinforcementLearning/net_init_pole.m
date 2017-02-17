function net = net_init()
% CNN_MNIST_LENET Initialize a CNN similar for MNIST


rng('default');
rng(0) ;

f=1/100 ;
net.layers = {} ;
%%linear
%{
%net.layers{end+1} = struct('type', 'bn') ;
net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*randn(2,4, 'single'), zeros(2,1,'single')}}) ;
%}

% 2-layer net
%
%net.layers{end+1} = struct('type', 'bn') ;
net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*randn(16,4, 'single'), zeros(16,1,'single')}}) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*randn(2,16, 'single'), zeros(2,1,'single')}}) ;
%}


%%leave it like this, we will evaluate the cost and derivative in another function  :p


