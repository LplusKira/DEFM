function ae = aesetopt(ae, opt)
	
	ae.alpha = opt.alpha;
	ae.wd = opt.wd;
	ae.beta = opt.beta;
	ae.rho = opt.rho;
	ae.momentum = opt.momentum;
	
	ae.inputZeroMaskedFraction = opt.inputZeroMaskedFraction;
	ae.dropoutFraction = opt.dropoutFraction;
	ae.Gaussian_Noise = opt.Gaussian_Noise;
	
	ae.unit = opt.unit;
	ae.output = opt.output;
end