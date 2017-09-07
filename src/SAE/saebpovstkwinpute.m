% stack autoencoder backpropagation over stacks with input error
function sae = saebpovstkwinpute(sae, e)
	oe = e;
	% backprop error from the input stack
	for i = 1 : numel(sae.ae)-1
		ae = sae.ae{i};
		ae.e = e;
		[ae, d] = nnbpgrads(ae);
		sae.ae{i} = ae;
		% backprop the error singal to the next stack
		%e = -d{2} * sae.ae{i+1}.W{2};
		e = -d{3} * ae.W{2};
	end
	ae = sae.ae{numel(sae.ae)};
	ae.e = e;
	[ae, d] = nnbpgrads(ae);
	sae.ae{numel(sae.ae)} = ae;
	e = oe;
end