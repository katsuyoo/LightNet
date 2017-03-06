function [opts]=test_net(net,opts)

    opts.training=0;

    opts.MiniBatchError=[];
    opts.MiniBatchError_Top5=[];
    opts.MiniBatchLoss=[];

    for mini_b=1:opts.n_test_batch
        
        idx=1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size;

        if length(size(opts.test))==2%%test mlp                    
            res(1).x=opts.test(:,idx);
        else %test cnn
             res(1).x=opts.test(:,:,:,idx);
        end
        
        res(1).class=opts.test_labels(idx);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%forward%%%%%%%%%%%%%%%%%%%
        [ net,res,opts ] = net_ff( net,res,opts );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    
        err=error_multiclass(res(1).class,res);    
        opts.MiniBatchError=[opts.MiniBatchError;err(1)/opts.parameters.batch_size];
        opts.MiniBatchError_Top5=[opts.MiniBatchError_Top5;err(2)/opts.parameters.batch_size];
        opts.MiniBatchLoss=[opts.MiniBatchLoss;gather(mean(res(end).x(:)))]; 
      
    end
    
    opts.results.TestEpochLoss=[opts.results.TestEpochLoss;mean(opts.MiniBatchLoss(:))];
    opts.results.TestEpochError=[opts.results.TestEpochError;mean(opts.MiniBatchError(:))];
    opts.results.TestEpochError_Top5=[opts.results.TestEpochError_Top5;mean(opts.MiniBatchError_Top5(:))];

    if isfield(opts,'test_labels')
         disp(['Epoch ',num2str(opts.parameters.current_ep),...
             ', testing loss: ', num2str(opts.results.TestEpochLoss(end)),...
                    ', top 1 err: ', num2str(opts.results.TestEpochError(end)),...
                    ',top 5 err:,',num2str(opts.results.TestEpochError_Top5(end))])                
    end
      
end


