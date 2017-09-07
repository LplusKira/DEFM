function nn = nntrain(nn, x, y, opts,TX,TY)
    assert(isfloat(x), 'x must be a float');
    m = size(x, 1);
    batchsize = opts.batchsize;
    numepochs = opts.numepochs;

    numbatches = m / batchsize;
    for i = 1 : numepochs
        tic;
        Loss = 0;
        kk = randperm(m);
        for l = 1 : numbatches
            batch_x = x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
            %add noise to input (for use in denoising autoencoder)
            if(nn.inputZeroMaskedFraction ~= 0)
                batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
            end
            
            batch_y = y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
            
            nn = nnff(nn, batch_x, batch_y);
            nn = nnbp(nn);            
            nn = nnapplygrads(nn);
            Loss = Loss + nn.L;
        end
        nn.alpha = nn.alpha*0.995;
        nn.momentum = nn.momentum*1.01;
        if nn.momentum >=0.8
         nn.momentum = 0.8;
        end
           t = toc;
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mean squared error is ' num2str(Loss/numbatches)])
        disp(['alpha is ' ,num2str(nn.alpha),' lambda is ',num2str(nn.lambda) , '  momentum is ',num2str(nn.momentum)]);
    end
    nn.L = Loss/numbatches;
end

