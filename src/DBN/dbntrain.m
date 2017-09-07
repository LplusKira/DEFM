function [dbn ] = dbntrain( dbn , train_x, opt )
    n_rbm = numel(dbn.rbm);
    GAUSSIAN = opt.GAUSSIAN;
    x = train_x;
    LR = opt.lr;
    if GAUSSIAN == 1
        opt.lr = 0.005;
    end
    dbn.rbm{1} = rbmtrain(dbn.rbm{1},x,opt,GAUSSIAN);
    x = sigm(x * dbn.rbm{1}.W + repmat(dbn.rbm{1}.bh , size(x,1),1));
    opt.lr = LR;    
    for i = 2 : n_rbm
        dbn.rbm{i} = rbmtrain(dbn.rbm{i},x,opt,0);
        x = sigm(x * dbn.rbm{i}.W + repmat(dbn.rbm{i}.bh , size(x,1),1));
    end

end
