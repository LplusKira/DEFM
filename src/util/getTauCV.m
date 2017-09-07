function [ tau_cv ] = getTauCV( cv_pred , X,Y,kk,totalCV )
TAU = 0:1/500:1;
MI_MAX = -100;
tau_cv = 0;
for t = TAU
    mi_sum = 0;
    for cv = 1 : totalCV
        [ ~,~,test_x,test_y ] = generateCVSet( X,Y,kk,cv, totalCV);
        pred = cv_pred{cv};
        mi = f_score(pred >= t , test_y);
        mi_sum = mi_sum + mi;
    end
    if MI_MAX < mi_sum
        MI_MAX  = mi_sum;
        tau_cv = t;
    end
end
end

