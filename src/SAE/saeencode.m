function [sae,code] = saeencode(sae, x)
	for i = 1 : numel(sae.ae)
		ae = sae.ae{i};
		t = nnff(ae, x, x);
		x = t.a{2};
		% remove bias term
		%x = x(:, 2:end);
		sae.ae{i} = t;
	end
	code = repmat(x, 1);
end