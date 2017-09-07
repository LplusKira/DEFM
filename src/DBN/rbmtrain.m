function [ rbm ] = rbmtrain( rbm,X,opt ,GAUSSIAN)
m = size(X,1);
max_iter = opt.max_iter;
b_s = opt.b_s;
n_b = ceil(m/b_s);
max_cd = opt.max_cd;
lr = opt.lr;
wd = opt.wd;
momen = opt.momen;
rho = opt.rho;
beta = opt.beta;
t = lr/max_iter;
%%%%%%%%%%%%%%%%%%%%%
for iter = 1 : max_iter
    kk = randperm(m);
    momen = momen * 1.05;
    if momen > 0.9
        momen = 0.9;
    end
    recon  =0;
    fe_dif = 0;
    recon_v = 0;
    
    for b = 1 : n_b
        x1 = X( kk ( (b-1) * b_s + 1 : min( m , b * b_s)) , : );
        batch_size = size(x1,1);
        h1 = sigm(x1 * rbm.W +repmat(rbm.bh , batch_size , 1));
        h_sample = h1 > rand(size(h1));
        %%%%%%%%%%%%%%% contrastive divergence
        for cd = 1 : max_cd
            if GAUSSIAN == 1
                x2 =(h_sample * rbm.W' + repmat(rbm.bv , batch_size , 1));
                x_sample = x2 + 0.05*randn(size(x2));
            else
                x2 =sigm(h_sample * rbm.W' + repmat(rbm.bv , batch_size , 1));
                x_sample = x2 > rand(size(x2));
            end
            h2 = sigm(x_sample * rbm.W + repmat(rbm.bh , batch_size , 1));
            h_sample = h2 > rand(size(h2));
        end
        s_grad = (mean(h1,1) - rho);
        %%%%%%%%%%%%%%%%
        rbm.dW = momen * rbm.dW + lr * ((x1'*h1 -x2'*h2)./batch_size - wd * rbm.W);
        rbm.dbv = momen * rbm.dbv + lr * mean(x1 - x2);
        rbm.dbh = momen * rbm.dbh + lr * (mean(h1 - h2) - beta* (s_grad.*mean(h1)));
        %%%%%%%%%%%%%%%%
        rbm.W = rbm.W + rbm.dW;
        rbm.bv = rbm.bv + rbm.dbv;
        rbm.bh = rbm.bh + rbm.dbh;
        recon = recon + sum(sum((x1-x2).^2));
    end
    %    lr = lr - t;
    
    if mod(iter , 1) == 0
        fprintf('learning RBM %d/%d, recon = %.6f\n',iter,max_iter,recon/m);        
    end
    
end



end

