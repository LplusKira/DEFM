function sae = saesetopt(sae, opt)
	for i = 1 : numel(sae.ae)
	
		sae.ae{i}.alpha = opt.alpha;
		sae.ae{i}.wd = opt.wd;
		sae.ae{i}.beta = opt.beta;
		sae.ae{i}.rho = opt.rho;
		sae.ae{i}.momentum = opt.momentum;
		
		sae.ae{i}.inputZeroMaskedFraction = opt.inputZeroMaskedFraction;
		sae.ae{i}.dropoutFraction = opt.dropoutFraction;
		sae.ae{i}.Gaussian_Noise = opt.Gaussian_Noise;
		
		sae.ae{i}.unit = opt.unit;
		sae.ae{i}.output = opt.output;
		
	end
end