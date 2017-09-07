function net = nnbp(net)
n = net.n;
%%  backprop
switch net.output
    case 'sigm'
        d{n} =  -net.e .* (net.a{n} .* (1 - net.a{n}));
    case 'ReLU'
        d{n} =  -net.e .* (net.a{n} > 0.0);
    case {'linear', 'softmax'}
        d{n} = -net.e;
end

%fprintf('Debug: size(d{n}) = ');
%disp(size(d{n}));
%fprintf('\n');

for i = (n - 1) : -1 : 2
    switch net.unit
        case 'sigm'
            d{i} = (d{i + 1} * net.W{i}) .* (net.a{i} .* (1 - net.a{i}));
        case 'ReLU'
            d{i} = (d{i + 1} * net.W{i}) .* (net.a{i} > 0.0);
        case 'linear'
            d{i} = d{i + 1} * net.W{i};
    end
    %    d{i} = (d{i + 1} * net.W{i}) .* (net.a{i} .* (1 - net.a{i}));
end

for i = 1 : (n - 1)
    net.prev_dW{i} = net.dW{i};
    net.prev_db{i} = net.db{i};
    net.dW{i} = net.dW{i} * net.momentum + net.alpha * (d{i + 1}' * net.a{i}) / size(d{i + 1}, 1) + net.alpha * net.wd * net.W{i};
    net.db{i} = net.db{i} * net.momentum + net.alpha * sum(d{i + 1}, 1)' / size(d{i + 1}, 1);
end
end
