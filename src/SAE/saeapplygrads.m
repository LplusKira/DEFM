function sae = saeapplygrads(sae)
	for i = 1 : numel(sae.ae)
		sae.ae{i} = nnapplygrads(sae.ae{i});
	end
end