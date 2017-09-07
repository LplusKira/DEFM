function [micro ] = f_score(a,b)
F = 0;
tp_sum = 0;
fp_sum =0 ;
fn_sum = 0;
for l =1:size(a,2)
    tp = 0; fp = 0; fn = 0; tn = 0;
    for i=1:size(a,1)
        if a(i,l) == 1 && b(i,l) == 1
            tp = tp +1;
        elseif a(i,l) == 1 && b(i,l) == 0
            fp = fp +1;
        elseif a(i,l) == 0 && b(i,l) == 1
            fn = fn+1;
        else
            tn = tn +1;
        end
        
        
    end
    if tp~=0 || fp~=0 || fn~=0
        F =  F +(2*tp)/(2*tp+fp+fn);
    end
    tp_sum = tp_sum + tp;
    fp_sum = fp_sum + fp;
    fn_sum = fn_sum + fn;
    
end
P = tp_sum/(tp_sum + fp_sum);
R = tp_sum/(tp_sum + fn_sum);

micro = 2*P*R/(P+R);
end