function [ prec ] = avg_precision( pred,true )
    [C,D] = sort(pred,2,'descend');
    Pos = zeros(size(pred,1) ,1);
    m = size(pred,1); d=  size(pred,2);
    prec = 0;
    for i=1:m
        score=  0;
        
        a  = find(true(i,:) == 1);
        for j=1:numel(a)
            temp = 0;
            for k=1:numel(a)
                if pred(i,a(k)) >= pred(i,a(j))
                  temp = temp+1;
                end
            end
            
            
            score = score + temp/find(D(i,:) == a(j));
        end
        prec = prec + score/numel(a);
    end
    prec = prec/m;

end

