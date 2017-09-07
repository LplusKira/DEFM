function net = nnapplygrads(net)
%  TODO add momentum
%    for i = 1 : (numel(net.size) - 1)
% for i = 1 : (net.n - 1)
%     %         net.g = sign(net.dW{i}).*sign(net.prev_dW{i});
%     %        net.g(find(net.g >0)) = 1.2;
%     %         net.g(find(net.g <=0)) = 0.5;
%     %        net.dW{i} = net.dW{i}.*net.g;
%     %%momentum
%     %       net.MSW{i} = 0.9*net.MSW{i} +0.1*net.dW{i}.^2;
%     %      net.MSb{i} = 0.9*net.MSb{i} +0.1*net.db{i}.^2;
%     %  net.dW{i} = net.momentum*net.prev_dW{i} - net.dW{i};
%     %  net.db{i} = net.momentum*net.prev_db{i} - net.db{i};
%     %% update
%     
% end
%%%% tied weight or not
for i=1:(net.n-1)
    %   net.W{i} = net.W{i} + net.alpha * net.dW{i}./sqrt(net.MSW{i}) - net.lambda * net.W{i};
    %   net.b{i} = net.b{i} + net.alpha * net.db{i}./sqrt(net.MSb{i});
    net.W{i} = net.W{i} - net.dW{i};
    net.b{i} = net.b{i} - net.db{i};
end
end
