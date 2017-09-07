function [ nn ] = dbntonn( dbn)
    nn = nnsetup(dbn.sizes);
    for i = 1 : nn.n - 1
        nn.W{i} = dbn.rbm{i}.W';
        nn.b{i} = dbn.rbm{i}.bh';
    end
end

