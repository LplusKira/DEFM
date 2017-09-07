function [sae,nn,crbm] = NCEMSAEjointTrain( sae,nn,crbm,X,Y,opt ,TX,TY,sae_opt)
assert(nn.size(end) == size(crbm.W,1));
m = size(X,1);
tm = size(TX,1);
max_iter = opt.max_iter;
b_s = opt.b_s;
n_b = ceil(m/b_s);
tn_b = ceil(tm/b_s);
max_cd = opt.max_cd;
lr = opt.lr;
wd = opt.wd;
momen = opt.momen;
t = lr/max_iter;
v = size(X,2);
sn = numel(sae.ae);
%sae_output = sae_opt.output;

%%%%%%%%%%%%%%%%%%%%%
for iter = 1 : max_iter
    kk = randperm(m);
    tkk = randperm(tm);
    momen = momen * 1.05;
    if momen > 0.9
        momen = 0.9;
    end
    recon  =0;
    fe_dif = 0;
    recon_v = 0;
    max_cd = ceil(iter/20);
    for b = 1 : n_b
        tb = rem(b, tn_b) + 1;
        %disp(tb);
        x1 = X( kk ( (b-1) * b_s + 1 : min( m , b * b_s)) , : );
        y1 = Y( kk ( (b-1) * b_s + 1 : min( m , b * b_s)) , : );
        tx1 = TX( tkk ( (tb-1) * b_s + 1 : min( tm , tb * b_s)) , : );
        allx1 = [x1; tx1];
        %% forward pass to SAE
        [sae, allcx1] = saeencode(sae, allx1);
        batch_size = size(x1, 1);
        cx1 = allcx1(1:batch_size, :);						% choose the code of training batch
        %% forward pass to NCEM
        f_x1 = ff( nn,cx1 );
        %batch_size = size(cx1,1);
        %disp(size(cx1));
        %disp(size(y1));
        %disp(size(f_x1));
        %disp(size(crbm.W));
        %disp(size(crbm.U));
        %disp(size(crbm.bh));
        h1 = sigm(f_x1 * crbm.W + y1 * crbm.U + repmat(crbm.bh , batch_size , 1));
        h_sample = double(h1 > rand(size(h1)));
        %%%%%%%%%%%%%%% Gibbs
        for cd = 1 : max_cd
            y2 = sigm( h_sample * crbm.U' + f_x1 * crbm.L + repmat(crbm.by , batch_size,1));
            y_sample = y2 > rand(size(y2));
            h2 = sigm(f_x1 * crbm.W + y_sample * crbm.U + repmat(crbm.bh , batch_size , 1));
            h_sample = h2 > rand(size(h2));
        end
        nn = nnfftocrbm(nn, cx1,y1,y2, crbm );
        %% backprop pass of NN and SAE
        [nn, dx] = nnbpgrads(nn);
        %sae.ae{1}.output = sae_opt;
        sae = saebpovstkwinpute(sae, sae.ae{1}.e);	
        %% Update NN and SAE
        nn = nnapplygrads(nn);
        sae = saeapplygrads(sae);
        %% backprop pass of SAE with grads from NCEM
        %sae.ae{sn}.output = 'linear';
        for i = 1 : numel(sae.ae)
            % remain the activation of training data
            sae.ae{i}.a{1} = sae.ae{i}.a{1}(1:batch_size, :);
            sae.ae{i}.a{2} = sae.ae{i}.a{2}(1:batch_size, :);
            sae.ae{i}.a{3} = sae.ae{i}.a{3}(1:batch_size, :);
            % remove the gradients info of input
            sae.ae{i}.dW{1} = zeros(size(sae.ae{i}.W{1}, 1), size(sae.ae{i}.W{1}, 2));
            sae.ae{i}.dW{2} = zeros(size(sae.ae{i}.W{2}, 1), size(sae.ae{i}.W{2}, 2));
        end
        sae = saebpovstkwcodee(sae, -dx{2} * nn.W{1});
        %% Update SAE with grads from NCEM
        sae = saeapplygrads(sae);
        %sae.ae{sn}.output = sae_output;
        
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
