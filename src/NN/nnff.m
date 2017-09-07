function net = nnff(net, x, y)
	n = net.n;
	m = size(x, 1);
	net.a{1} = x;
	%% feedforward pass
	for i = 2 : n-1
		switch net.unit
			case 'sigm'
				net.a{i} = sigm(repmat(net.b{i - 1}', m, 1) + net.a{i - 1} * net.W{i - 1}');
			case 'ReLU'
				net.a{i} = max( 0 , repmat(net.b{i - 1}', m, 1) + net.a{i - 1} * net.W{i - 1}' );
		end
		if(net.dropoutFraction > 0 && i<n)
			if(net.testing)
				net.a{i} = net.a{i}.*(1 - net.dropoutFraction);
			else
				net.a{i} = net.a{i}.*(rand(size(net.a{i}))>net.dropoutFraction);
			end
		end
	end
	
	switch net.output
		case 'sigm'
			net.a{n} = sigm(repmat(net.b{n - 1}', m, 1) + net.a{n - 1} * net.W{n - 1}');
		case 'ReLU'
			net.a{n} = max( 0 , repmat(net.b{n - 1}', m, 1) + net.a{n - 1} * net.W{n - 1}' );
		case 'linear'
			net.a{n} = repmat(net.b{n - 1}', m, 1) + net.a{n - 1} * net.W{n - 1}';
		case 'softmax'
			net.a{n} = repmat(net.b{n - 1}', m, 1) + net.a{n - 1} * net.W{n - 1}';
			net.a{n} = exp(bsxfun(@minus, net.a{n}, max(net.a{n}, [], 2)));
			net.a{n} = bsxfun(@rdivide, net.a{n}, sum(net.a{n}, 2));
	end
	
	% error & loss
	net.e = y - net.a{n};
	
	switch net.output
		case {'sigm', 'linear', 'ReLU'}
			e = y - net.a{n};
			net.L = 1/2 * sum(sum(e .^ 2)) / m;
		case 'softmax'
			net.L = -sum(sum(y .* log(net.a{n}))) / m;
	end
end