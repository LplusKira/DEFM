function [ae, code] = aeencode(ae, x)
	n = ae.n;
	L = ceil(n/2);
	ae = nnff(ae, x, x);
	code = ae.a{L};
end