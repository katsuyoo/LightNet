files=dir([fullfile(opts.output_dir,opts.output_name)]);

if opts.LoadNet && length(files)>1
    [~,last_file]=sort([files(:).datenum],'descend');
    if length(files)<opts.n_epoch,end  
    load(fullfile(opts.output_dir,files(last_file(1)).name));
    opts.parameters=parameters;
    opts.results=results;
    
    opts.parameters.current_ep=opts.parameters.current_ep+1;
end

if opts.LoadNet==0 || length(files)==0  
    
    net=NetInit(opts);
    opts.results=[];
    opts.results.TrainEpochError=[];
    opts.results.TestEpochError=[];
    opts.results.TrainEpochError_Top5=[];
    opts.results.TestEpochError_Top5=[];
    opts.results.TrainEpochLoss=[];
    opts.results.TestEpochLoss=[];
    opts.results.TrainLoss=[];
    opts.results.TrainError=[];
    opts.results.TrainError_Top5=[];
end

opts.RecordStats=0;

opts.n_batch=floor(opts.n_train/opts.parameters.batch_size);
opts.n_test_batch=floor(opts.n_test/opts.parameters.batch_size);


if(opts.use_gpu)       
    for i=1:length(net)
        net(i)=SwitchProcessor(net(i),'gpu');
    end
else
    for i=1:length(net)
        net(i)=SwitchProcessor(net(i),'cpu');
    end
end

start_ep=opts.parameters.current_ep;
if opts.plot
    figure1=figure;
end
for ep=start_ep:opts.n_epoch
    
    
    [net,opts]=train_net(net,opts);  
    [opts]=test_net(net,opts);
    
    if opts.plot
        subplot(1,2,1); plot(opts.results.TrainEpochError,'b','DisplayName','Train (top1)');hold on;plot(opts.results.TestEpochError,'r','DisplayName','Test (top1)');hold on;
        plot(opts.results.TrainEpochError_Top5,'b--','DisplayName','Train (top5)');plot(opts.results.TestEpochError_Top5,'r--','DisplayName','Test (top5)');hold off;
        title('Error Rate per Epoch');legend('show');
        subplot(1,2,2); plot(opts.results.TrainEpochLoss,'b','DisplayName','Train');hold on;plot(opts.results.TestEpochLoss,'r','DisplayName','Test');hold off;
        title('Loss per Epoch');legend('show')
        drawnow;
        saveas(gcf,[fullfile(opts.output_dir2,[opts.output_name2,num2str(ep),'.pdf'])])
    end
    
    parameters=opts.parameters;
    results=opts.results;
    save([fullfile(opts.output_dir2,[opts.output_name2,num2str(ep),'.mat'])],'net','parameters','results');   
    opts.parameters.current_ep=opts.parameters.current_ep+1;
    
end

opts.train=[];
opts.test=[];

[min_err,min_id]=min(opts.results.TestEpochError);
disp(['Lowest error rate: ',num2str(min_err)]);
best_net_source=[fullfile(opts.output_dir2,[opts.output_name2,num2str(min_id),'.mat'])];
best_net_destination=[fullfile(opts.output_dir2,['best_',opts.output_name2,'.mat'])];
copyfile(best_net_source,best_net_destination);


