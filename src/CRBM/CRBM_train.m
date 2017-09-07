function [ crbm ] = CRBM_train( crbm , V , Y,opt)
%%%%%%%%%%%%%%%%%%%%%
fprintf('Learning conditional restricted Boltzmann machine....\n');
max_iter = opt.max_iter;
b_s = opt.b_s;
m = size(V,1);
n_b = ceil(m/b_s);
max_cd = opt.max_cd;
lr = opt.lr;
wd = opt.wd;
momen = 0.5;
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
        v1 = V( kk ( (b-1) * b_s + 1 : min( m , b * b_s)) , : );
        y1 = Y( kk ( (b-1) * b_s + 1 : min( m , b * b_s)) , : );
        batch_size = size(v1,1);
        h1 = sigm((v1 * crbm.W + y1 * crbm.U) + repmat(crbm.bh , batch_size , 1));
        h_sample = double(h1 > rand(size(h1)));
        %%%%%%%%%%%%%%% contrastive divergence
        for cd = 1 : max_cd
            y2 = sigm( h_sample * crbm.U' + v1 * crbm.L + repmat(crbm.by , batch_size,1));
            y_sample = y2 > rand(size(y2));
            h2 = sigm(v1 * crbm.W + y_sample * crbm.U + repmat(crbm.bh , batch_size , 1));
            h_sample = h2 > rand(size(h2));
%             [ fe ] = CRBM_getFreeEnergy( crbm,v1,y2 );
%             ind = find(fe < MINFE);
%             MINFE(ind,:) = fe(ind,:);
%             y(ind,:) = y2(ind,:);
        end
      %  y2 = y;
      %  h2 = sigm(v1 * crbm.W + y2 * crbm.U + repmat(crbm.bh , batch_size , 1));

        crbm.dW = momen * crbm.dW + lr * ((v1'*(h1 -h2)./batch_size) - wd * crbm.W);
        crbm.dU = momen * crbm.dU + lr * ((y1'*h1 - y2'*h2)./batch_size - wd * crbm.U);
        crbm.dL = momen * crbm.dL + lr * ((v1'*(y1 -y2)./batch_size) - wd * crbm.L);
        crbm.dbh = momen * crbm.dbh + lr * mean(h1 - h2);
        crbm.dby = momen * crbm.dby + lr * mean(y1 - y2);
        %%%%%%%%%%%%%%%%
        crbm.W = crbm.W + crbm.dW;
        crbm.U = crbm.U + crbm.dU;
        crbm.L = crbm.L + crbm.dL;
        crbm.bh = crbm.bh + crbm.dbh;
        crbm.by = crbm.by + crbm.dby;
        recon = recon + sum(sum((y1-y2).^2));
    end
    if mod(iter , 1) == 0
        fprintf('learning CRBM %d/%d, recon y = %.6f\n',iter,max_iter,recon/m);
    end    
end
end
