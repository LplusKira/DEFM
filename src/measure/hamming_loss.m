function [ loss ] = hamming_loss( pred,true)
	m = size(pred,1); d = size(pred,2);
	loss = 0;
	for i=1:m
		pred_bin = (pred(i,:) >= 0.5);
		true_bin = (true(i,:) >= 0.5);
		diff = (pred_bin ~= true_bin);
		loss = loss + sum(diff)/d;
	end
	loss = loss/m;
end