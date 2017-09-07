function [pred] = NCEMTestGibbs(NN,CRBM,X,Y,g_step)
    if nargin == 4
        g_step = 200;
    end
    pred = zeros(size(Y));
    NN.testing = 1;
    X = ff(NN,X);    
    [ pred ] = CRBM_testMLC( CRBM , X,Y  );

end

