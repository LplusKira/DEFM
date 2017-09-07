% stack autoencoder backpropagation over stacks with code error
function sae = saebpovstkwcodee(sae, e)
	oe = e;
	% backprop error from the code stack	
	for i = numel(sae.ae) : -1 : 1
		ae = sae.ae{i};
		
		switch ae.unit
			case 'sigm'
				d{2} = -e .* (ae.a{2} .* (1 - ae.a{2}));
			case 'ReLU'
				d{2} = -e .* (ae.a{2} > 0.0);
			case 'linear'
				d{2} = -e;
		end
		
		% we only use the error of code to adjust the weight of encoder of AE !!
		d{3} = zeros(size(ae.e, 1), size(ae.e, 2));		
		
		ae.prev_dW{1} = ae.dW{1};
		ae.prev_db{1} = ae.db{1};
		ae.dW{1} = ae.dW{1} * ae.momentum + ae.alpha * (d{2}' * ae.a{1}) / size(d{2}, 1) + ae.alpha * ae.wd * ae.W{1};
		ae.db{1} = ae.db{1} * ae.momentum + ae.alpha * sum(d{2}, 1)' / size(d{2}, 1);
		
		sae.ae{i} = ae;
		% backprop the error singal to the next stack
		e = -d{2} * ae.W{1};
	end
	e = oe;
end