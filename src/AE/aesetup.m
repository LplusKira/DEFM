function ae = aesetup(size)
	L = numel(size);
	ae = nnsetup([size(1:L-1), fliplr(size)]);
end