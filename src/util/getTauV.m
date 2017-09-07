function [ tau_cv ] = getTauV( cv_pred , X,Y,kk,totalCV,testCV )
TAU = 0:1/500:1;
MI_MAX = -100;
tau_cv = 0;
y = size(Y,2);
for t = TAU
    mi_sum = 0;
    for cv = 1 : testCV
        [ ~,~,test_x,test_y ] = generateCVSet( X,Y,kk,cv, totalCV);
        pred = cv_pred{cv};
        % convert to binary label
        pred_bin = pred(:,1:2:y) >= pred(:,2:2:y);
        test_y_bin = test_y(:,1:2:y) >= test_y(:,2:2:y);
        mi = f_score(pred_bin >= t , test_y_bin);
        mi_sum = mi_sum + mi;
    end
    if MI_MAX < mi_sum
        MI_MAX  = mi_sum;
        tau_cv = t;
    end
end
end

