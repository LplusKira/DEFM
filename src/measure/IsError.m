function [ loss] = IsError( pred,true )
    [C,D] = sort(pred,2,'descend');
    Pos = zeros(size(pred,1) ,1);
    m = size(pred,1); d=  size(pred,2);
    incor = 0;
    for i=1:m  
        if true(i,D(i,1)) ~= 1          
            incor = incor +1;
        end
    end
    loss = incor/m;

end

