function [rl,cov,pre,one,mi_f,hl] = get_all_measure(pred ,  test_y , tau); 
assert(nargin == 3);

rl = rankingloss(pred,test_y);
cov = coverage(pred,test_y);
pre = avg_precision(pred,test_y);
one = IsError(pred,test_y);
[mi_f] = f_score(pred>=tau,test_y);
hl = hamming_loss(pred,test_y);

end

