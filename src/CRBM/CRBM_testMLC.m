function [ pred] = CRBM_testMLC( crbm,  V , Y,max_gibbs)

m = size(V,1);
burn_in = 100;
max_gibbs = 2000;
pred = zeros(size(Y));
y = zeros(size(Y));
C = 0;
%%%%%%%%%%%%%%%%%%%%%% burn_in Gibbs
for i = 1 : burn_in
    h = sigm( (V * crbm.W + y * crbm.U) + repmat(crbm.bh , m , 1));
    h = double(h > rand(size(h)));
    y = sigm( (h * crbm.U' + V * crbm.L) + repmat(crbm.by , m , 1));
    y = double(y > rand(size(y)));
end
%%%%%%%%%%%%%%%%%%%% Gibbs
for cd = 1 : max_gibbs
    h = sigm( (V * crbm.W + y * crbm.U) + repmat(crbm.bh , m , 1));
    h = double(h > rand(size(h)));
    y = sigm( (h * crbm.U' + V * crbm.L) + repmat(crbm.by , m , 1));
    if mod(cd,10) == 0
        pred = pred + y;
        C = C + 1;
    end
    y = double(y > rand(size(y)));
end
pred = pred./C;
end

