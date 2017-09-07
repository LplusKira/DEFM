function [ f_x] = ff( net,x )
n = net.n;
m = size(x, 1);
net.a{1} = x ;
%%  feedforward pass
for i = 2 : n
    net.a{i} = sigm(repmat(net.b{i - 1}', m, 1) + net.a{i - 1} * net.W{i - 1}');   
    if(net.dropoutFraction > 0 && i<n)
        if(net.testing)
            net.a{i} = net.a{i}.*(1 - net.dropoutFraction);
        else
            net.a{i} = net.a{i}.*(rand(size(net.a{i}))>net.dropoutFraction);
        end
    end
end
f_x = net.a{n};

end

