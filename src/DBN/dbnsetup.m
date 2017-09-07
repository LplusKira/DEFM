function [dbn] = dbnsetup(x , hidden)

dbn.rbm{1}.W = 0.1*randn(x , hidden(1));
dbn.rbm{1}.bh = zeros(1 , hidden(1));
dbn.rbm{1}.bv = zeros(1 , x);
dbn.rbm{1}.dW =  zeros(x , hidden(1));
dbn.rbm{1}.dbh = zeros(1 , hidden(1));
dbn.rbm{1}.dbv = zeros(1 , x);
dbn.sizes = [x,hidden];
for i = 2 : numel(hidden)
    dbn.rbm{i}.W = 0.1*randn(hidden(i-1) , hidden(i));
    dbn.rbm{i}.bh = zeros(1 , hidden(i));
    dbn.rbm{i}.bv = zeros(1 , hidden(i-1));
    dbn.rbm{i}.dW =  zeros(hidden(i-1) , hidden(i));
    dbn.rbm{i}.dbh = zeros(1 , hidden(i));
    dbn.rbm{i}.dbv = zeros(1 , hidden(i-1));
    
end

end

