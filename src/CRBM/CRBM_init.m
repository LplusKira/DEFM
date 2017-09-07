function [ crbm ] = CRBM_init( v , h , y )
%%%%%%%%%%%   init parameters
        crbm.W = 0.05*randn(v,h);
        crbm.U = 0.05*randn(y,h);
        crbm.L = 0.05*randn(v,y);

        crbm.bh = zeros(1,h);
        crbm.bv = zeros(1,v);
        crbm.by = zeros(1,y);
%%%%%%%%%%%%   init delta parameters
        crbm.dW =  0*randn(v,h);
        crbm.dU =  0*randn(y,h);
        crbm.dL =  0*randn(v,y);
        crbm.dbh = zeros(1,h);
        crbm.dbv = zeros(1,v);
        crbm.dby = zeros(1,y);


end

