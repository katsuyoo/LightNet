function [net,opts]=train_net(net,opts)

    opts.training=1;

    if ~isfield(opts.parameters,'learning_method')
        opts.parameters.learning_method='sgd';        
    end
    
    if ~isfield(opts,'display_msg')
        opts.display_msg=1; 
    end
    opts.MiniBatchError=[];
    opts.MiniBatchError_Top5=[];
    opts.MiniBatchLoss=[];

    tic
    
    opts.order=randperm(opts.n_train);    
    if opts.parameters.selective_sgd==1         
         [ net,opts ] = selective_sgd( net,opts );
    end

    for mini_b=1:opts.n_batch
                
        idx=opts.order(1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size);
        
        if length(size(opts.train))==2%%train mlp                    
            res(1).x=opts.train(:,idx);
        end        
            
        if length(size(opts.train))>2%%train cnn
            res(1).x=opts.train(:,:,:,idx);
        end
                
        if isfield(opts,'train_labels')
            res(1).class=opts.train_labels(idx);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%forward%%%%%%%%%%%%%%%%%%%
        [ net,res,opts ] = net_ff( net,res,opts );    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%backward%%%%%%%%%%%%%%%%
        opts.dzdy=single(1.0);        
        [ net,res,opts ] = net_bp( net,res,opts );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %%summarize the current batch
        loss=gather(mean(res(end).x(:)));
        err=error_multiclass(res(1).class,res);
        opts.MiniBatchError=[opts.MiniBatchError;err(1)/opts.parameters.batch_size];
        opts.MiniBatchError_Top5=[opts.MiniBatchError_Top5;err(2)/opts.parameters.batch_size];
        if opts.display_msg==1
            disp(['Minibatch loss: ', num2str(loss),...
                ', top 1 err: ', num2str(opts.MiniBatchError(end)),...
                ',top 5 err:,',num2str(opts.MiniBatchError_Top5(end))])
        end
       
        opts.MiniBatchLoss=[opts.MiniBatchLoss;loss];                 
        if (~isfield(opts.parameters,'iterations'))
            opts.parameters.iterations=0; 
        end
        opts.parameters.iterations=opts.parameters.iterations+1;
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%stochastic gradients descent%%%%%%%%%%%%%%%%%%%%%%%%
        [ net,res,opts ] = opts.parameters.learning_method( net,res,opts );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    
    %%summarize the current epoch
    opts.results.TrainEpochError=[opts.results.TrainEpochError;mean(opts.MiniBatchError(:))];
    opts.results.TrainEpochError_Top5=[opts.results.TrainEpochError_Top5;mean(opts.MiniBatchError_Top5(:))];
    opts.results.TrainEpochLoss=[opts.results.TrainEpochLoss;mean(opts.MiniBatchLoss(:))];
    
    if isfield(opts,'train_labels')
         disp(['Epoch ',num2str(opts.parameters.current_ep),...
         ', training loss: ', num2str(opts.results.TrainEpochLoss(end)),...
                ', top 1 err: ', num2str(opts.results.TrainEpochError(end)),...
                ',top 5 err:,',num2str(opts.results.TrainEpochError_Top5(end))])                
    end

    if opts.RecordStats==1
        opts.results.TrainLoss=[opts.results.TrainLoss;opts.MiniBatchLoss];
        opts.results.TrainError=[opts.results.TrainError;opts.MiniBatchError]; 
        opts.results.TrainError_Top5=[opts.results.TrainError_Top5;opts.MiniBatchError_Top5]; 
    end
    
    toc;

end




