function net = nnfftocrbm(net, x,y1,y2, CRBM )
n = net.n;
m = size(x, 1);
net.a{1} = x ;
%%  feedforward pass
for i = 2 : n
    switch net.unit
        case 'sigm'
            net.a{i} = sigm(repmat(net.b{i - 1}', m, 1) + net.a{i - 1} * net.W{i - 1}');
        case 'ReLU'
            net.a{i} = max( 0 , repmat(net.b{i - 1}', m, 1) + net.a{i - 1} * net.W{i - 1}' ) ;
    end
    if(net.dropoutFraction > 0 && i<n)
        if(net.testing)
            net.a{i} = net.a{i}.*(1 - net.dropoutFraction);
        else
            net.a{i} = net.a{i}.*(rand(size(net.a{i}))>net.dropoutFraction);
        end
    end
    %toc;
end
f_x = net.a{net.n};
[ fe_pos ] = CRBM_getFreeEnergy( CRBM , f_x , y1 );
[ fe_neg ] = CRBM_getFreeEnergy( CRBM , f_x , y2 );

net.L =  sum(fe_pos - fe_neg) ./ m;
net.e = (CRBM.W * sigm(net.a{n}*CRBM.W + y1 * CRBM.U + repmat(CRBM.bh , m , 1))' +CRBM.L*y1')...
    -(CRBM.W * sigm(net.a{n}*CRBM.W + y2 * CRBM.U + repmat(CRBM.bh , m , 1))' +CRBM.L*y2');
net.e = net.e';
%fprintf('Debug: size(net.e)=');
%disp(size(net.e));
%fprintf('\n');

end
