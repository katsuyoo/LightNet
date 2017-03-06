function [ net,res,opts ] = net_ff( net,res,opts )
%NET_FF Summary of this function goes here
%   Detailed explanation goes here

    if ~isfield(opts,'datatype')
        opts.datatype='single';
    end
    
    res(1).x=cast(res(1).x,opts.datatype);
    
    if opts.use_gpu
        res(1).x=gpuArray(single(res(1).x));
    end
    
    for layer=1:numel(net.layers)

        opts.current_layer=layer;
        switch net.layers{layer}.type

            case {'conv' , 'conv2d'}
                if isfield(net.layers{1,layer},'pad')
                    if(length(net.layers{1,layer}.pad)==1)
                        net.layers{1,layer}.pad=ones(1,4)*net.layers{1,layer}.pad;
                    end
                else
                   net.layers{1,layer}.pad=[];
                end
                
                if isfield(net.layers{1,layer},'stride')
                    if(length(net.layers{1,layer}.stride)==1)
                        net.layers{1,layer}.stride=ones(1,2)*net.layers{1,layer}.stride;
                    end
                else
                   net.layers{1,layer}.stride=1;
                end
                
                [res(layer+1).x,~,~,opts] = conv_layer_2d( res(layer).x,net.layers{1,layer}.weights{1},net.layers{1,layer}.weights{2},net.layers{1,layer}.stride,net.layers{1,layer}.pad,[],opts );
                
            case {'mlp','linear'} 
                [res(layer+1).x,~,~,opts] = linear_layer( res(layer).x,net.layers{1,layer}.weights{1},net.layers{1,layer}.weights{2},[], opts);

            case 'dropout'
                if opts.training
                    dropout_opts.rate=net.layers{1,layer}.rate;
                    dropout_opts.mask=[];
                    [res(layer+1).x,dropout_opts.mask]= dropout(res(layer).x,[],dropout_opts );
                    net.layers{1,layer}.opts=dropout_opts;
                else
                    res(layer+1).x=res(layer).x;
                end
                
            case 'bnorm'
                [net,res(layer+1).x,~,~,opts] = bnorm( net,res(layer).x,layer,[],opts );
            case {'normalize', 'lrn'}
                [res(layer+1).x,opts] = lrn(res(layer).x, net.layers{1,layer}.param(1),net.layers{1,layer}.param(2),net.layers{1,layer}.param(3),net.layers{1,layer}.param(4),[],opts) ;
            
            case 'relu'
                res(layer+1).x = relu(res(layer).x,[] );
            case 'leaky_relu'
                res(layer+1).x = leaky_relu(res(layer).x,[] );
            case 'sigmoid'
                res(layer+1).x = sigmoid_ln(res(layer).x,[] );
            case 'tanh'
                res(layer+1).x = tanh_ln(res(layer).x,[] );
            
            case 'pad'
                res(layer+1).x = pad_data(res(layer).x,net.layers{1,layer}.pad,[]);

            case 'pool' 
                
                if isfield(net.layers{1,layer},'pad')
                    if(length(net.layers{1,layer}.pad)==1)
                        net.layers{1,layer}.pad=ones(1,4)*net.layers{1,layer}.pad;
                    end
                else
                   net.layers{1,layer}.pad=[];
                end
                
                if isfield(net.layers{1,layer},'stride')
                    if(length(net.layers{1,layer}.stride)==1)
                        net.layers{1,layer}.stride=ones(1,2)*net.layers{1,layer}.stride;
                    end
                end
                
                if opts.training==1
                    [res(layer+1).x,res(layer+1).from,opts] = maxpool(res(layer).x,net.layers{1,layer}.pool,net.layers{1,layer}.stride,net.layers{1,layer}.pad,[],[],opts);
                else
                    [res(layer+1).x,~,opts] = maxpool(res(layer).x,net.layers{1,layer}.pool,net.layers{1,layer}.stride,net.layers{1,layer}.pad,[],[],opts);
                end
            case 'softmaxloss'
                res(layer+1).x = softmaxlogloss(res(layer).x, res(1).class) ;               
            case 'softmax'        
                res(layer+1).x = softmax(res(layer).x,[]) ;
           
        end
    end

end

