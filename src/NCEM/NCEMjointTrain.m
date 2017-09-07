function [ nn,crbm] = NCEMjointTrain( nn,crbm,X,Y,opt ,TX,TY)
assert(nn.size(end) == size(crbm.W,1));
m = size(X,1);
max_iter = opt.max_iter;
b_s = opt.b_s;
n_b = ceil(m/b_s);
max_cd = opt.max_cd;
lr = opt.lr;
wd = opt.wd;
momen = opt.momen;
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
    max_cd = ceil(iter/20);
    for b = 1 : n_b
        x1 = X( kk ( (b-1) * b_s + 1 : min( m , b * b_s)) , : );
        y1 = Y( kk ( (b-1) * b_s + 1 : min( m , b * b_s)) , : );
        f_x1 = ff( nn,x1 );
        batch_size = size(x1,1);
        h1 = sigm(f_x1 * crbm.W + y1 * crbm.U + repmat(crbm.bh , batch_size , 1));
        h_sample = double(h1 > rand(size(h1)));
        %%%%%%%%%%%%%%% Gibbs
        for cd = 1 : max_cd
            y2 = sigm( h_sample * crbm.U' + f_x1 * crbm.L + repmat(crbm.by , batch_size,1));
            y_sample = y2 > rand(size(y2));
            h2 = sigm(f_x1 * crbm.W + y_sample * crbm.U + repmat(crbm.bh , batch_size , 1));
            h_sample = h2 > rand(size(h2));
        end
        nn = nnfftocrbm(nn, x1,y1,y2, crbm );
        nn = nnbp(nn);
        nn = nnapplygrads(nn);
        %%%%%%%%%%%%%%%%
        crbm.dW = momen * crbm.dW + lr * ((f_x1'*(h1 -h2)./batch_size) - wd * crbm.W);
        crbm.dU = momen * crbm.dU + lr * ((y1'*h1 - y2'*h2)./batch_size - wd * crbm.U);
        crbm.dL = momen * crbm.dL + lr * ((f_x1'*(y1 -y2)./batch_size) - wd * crbm.L);
        crbm.dbh = momen * crbm.dbh + lr * mean(h1 - h2);
        crbm.dby = momen * crbm.dby + lr * mean(y1 - y2);
        %%%%%%%%%%%%%%%%
        crbm.W = crbm.W + crbm.dW;
        crbm.U = crbm.U + crbm.dU;
        crbm.L = crbm.L + crbm.dL;
        crbm.bh = crbm.bh + crbm.dbh;
        crbm.by = crbm.by + crbm.dby;
        recon = recon + sum(sum((y1-y2).^2));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% learn NN
    end
%     if mod(iter , 10) == 0
%         [pred] = NCEMTestGibbs(nn,crbm,TX,TY,500);
%         rl = rankingloss(pred,TY)
%     end
    if mod(iter,1) == 0
        fprintf('fine-tune NCEM %d/%d, recon y = %.6f\n',iter,max_iter,recon/m);
    end
end

end
