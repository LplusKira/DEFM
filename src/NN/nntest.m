function [er, bad] = nntest(net, x, y)
    e = 0;
    bad = [];
    net.testing = 1;
    batch = int16(floor(size(x,1) / 10));
    for i = 1 : size(x,1)/batch + int16(mod(size(x,1) , batch) ~=0 )
        %  feedforward
        
        X = x((i-1)*batch + 1: min(i*batch,size(x,1)) , :);
        Y = y((i-1)*batch + 1: min(i*batch,size(x,1)) , :);
        net = nnff(net, X, Y);
        
        [~, g] = max(net.a{net.n} , [] , 2);
        [~,cmpY] = max(Y , [] , 2);    
        e = e + sum(g~=cmpY);
    end
    er = e / size(x, 1);
end
