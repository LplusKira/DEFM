% autoencoder backpropagation with code error
function ae = aebpwithcodee(ae, e)
	oe = e;
	en = ceil(ae.n/2);
	switch ae.unit
		case 'sigm'
			d{en} = -e .* (ae.a{en} .* (1 - ae.a{en}));
		case 'ReLU'
			d{en} = -e .* (ae.a{en} > 0.0);
		case 'linear'
			d{en} = -e;
	end
	
	for i = (en - 1) : -1 : 2
		switch ae.unit
			case 'sigm'
				d{i} = (d{i + 1} * ae.W{i}) .* (ae.a{i} .* (1 - ae.a{i}));
			case 'ReLU'
				d{i} = (d{i + 1} * ae.W{i}) .* (ae.a{i} > 0.0);
			case 'linear'
				d{i} = d{i + 1} * ae.W{i};
		end
	end
	
	for i = 1 : (en - 1)
		%size(ae.dW{i})
		%size(d{i+1})
		%size(ae.a{i})
		%size(ae.W{i})
		ae.dW{i} = ae.dW{i} * ae.momentum + ae.alpha * (d{i + 1}' * ae.a{i}) / size(d{i + 1}, 1) + ae.alpha * ae.wd * ae.W{i};
		ae.db{i} = ae.db{i} * ae.momentum + ae.alpha * sum(d{i + 1}, 1)' / size(d{i + 1}, 1);
	e = oe;
	end
end