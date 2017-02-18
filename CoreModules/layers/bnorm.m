function [ net,y,dzdw,dzdb,opts ] = bnorm( net,x,layer_idx,dzdy,opts )
%BNORM Summary of this function goes here
%   Detailed explanation goes here
    dzdw=[];
    dzdb=[];
    if ~isfield(net,'iterations')
        net.iterations=0;
    end
    if ~isfield(net,'iterations_bn')
        net.iterations_bn=0; 
    else
        net.iterations_bn=net.iterations;
    end
    
    if ~isfield(opts.parameters, 'eps_bn')
        opts.parameters.eps_bn=1e-2;
    end
    
    if ~isfield(opts.parameters, 'mom_bn')
        opts.parameters.mom_bn=0.999;
    end
        
    if ~isfield(opts.parameters, 'simple_bn')
        opts.parameters.simple_bn=0;
    end
    
    batch_dim=length(size(x));%% This assumes the batch size must be >1 
    shape_x=size(x);
    
    if batch_dim==4
       x=permute(x,[3,1,2,4]);x=reshape(x,size(x,1),[]);
    end
    if batch_dim==3 
       x=permute(x,[2,1,3]);x=reshape(x,size(x,1),[]);
    end

    if ~isfield(net.layers{1,layer_idx},'weights')        
        sz=[shape_x(end-1),1];
        net.layers{1,layer_idx}.weights{1}=ones(sz,'like',x);
        for i=2:4
            net.layers{1,layer_idx}.weights{i}=zeros(sz,'like',x);
        end
        for i=1:2
            net.layers{1,layer_idx}.momentum{i}=zeros(sz,'like',x);
        end        
    end
    
    if net.iterations_bn==0
        for i=3:4
            net.layers{1,layer_idx}.weights{i}=0;
        end
    end
    mom_factor=1-opts.parameters.mom_bn.^(net.iterations_bn+1);
    
    
    if(opts.training&&isempty(dzdy))
        net.layers{1,layer_idx}.weights{3}=opts.parameters.mom_bn*net.layers{1,layer_idx}.weights{3}+(1-opts.parameters.mom_bn)*mean(x,2);   
        net.layers{1,layer_idx}.weights{4}=opts.parameters.mom_bn*net.layers{1,layer_idx}.weights{4}+(1-opts.parameters.mom_bn)*mean(x.^2,2);  

    end
            
    if(isempty(dzdy))
        
        net.layers{layer_idx}.x_n=bsxfun(@minus,x,net.layers{1,layer_idx}.weights{3}./mom_factor);
        net.layers{layer_idx}.x_n=bsxfun(@rdivide,net.layers{layer_idx}.x_n,(net.layers{1,layer_idx}.weights{4}./mom_factor+opts.parameters.eps_bn).^0.5);
        
        y=bsxfun(@times,net.layers{layer_idx}.x_n,net.layers{1,layer_idx}.weights{1});
        y=bsxfun(@plus,y,net.layers{1,layer_idx}.weights{2});
        
        
    else
        
        if batch_dim==4
            dzdy=permute(dzdy,[3,1,2,4]);dzdy=reshape(dzdy,size(dzdy,1),[]);
        end
        if batch_dim==3 
            dzdy=permute(dzdy,[2,1,3]);dzdy=reshape(dzdy,size(dzdy,1),[]);
        end
        
        dzdw=mean(dzdy.*net.layers{layer_idx}.x_n,2);
        dzdb=mean(dzdy,2);
        
        if ~opts.parameters.simple_bn
            %the complicated version         
            tmp=bsxfun(@minus,x,net.layers{1,layer_idx}.weights{3}./mom_factor);
            tmp=bsxfun(@rdivide,tmp,(net.layers{1,layer_idx}.weights{4}./mom_factor+opts.parameters.eps_bn).^(0.5));
            tmp=bsxfun(@times,dzdw,tmp);
            tmp=bsxfun(@plus,dzdb,tmp);
            dzdy=(dzdy-tmp.*(1-opts.parameters.mom_bn));
            clear tmp;
            %[max(abs(dzdy(:))), max(abs(tmp(:)))]
        end
        
        %the simple version:
        y=bsxfun(@times,dzdy,net.layers{1,layer_idx}.weights{1});
        y=bsxfun(@rdivide,y,(net.layers{1,layer_idx}.weights{4}./mom_factor+opts.parameters.eps_bn).^(0.5));

    end
    
    if batch_dim==4
        y=reshape(y,shape_x(3),shape_x(1),shape_x(2),shape_x(4));
        y=permute(y,[2,3,1,4]);
    end
    if batch_dim==3 
        y=reshape(y,shape_x(2),shape_x(1),shape_x(3));
        y=permute(y,[2,1,3]);
    end
        
end
