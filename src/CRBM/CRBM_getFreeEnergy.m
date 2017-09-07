function [ fe ] = CRBM_getFreeEnergy( crbm,v,y )
    m = size(v,1);
    fe = -sum(log(1 + exp(repmat(crbm.bh,m,1) + v*crbm.W + y*crbm.U)) , 2) - y*crbm.by' - sum(v*crbm.L.*y,2);

end

