function [ loss ] = rankingloss( pred,true)
    [C,D] = sort(pred,2,'descend');
    Pos = zeros(size(pred,1) ,1);
    m = size(pred,1); d=  size(pred,2);
    loss = 0;
    for i=1:m
        temp = 0;
        incor = 0;
        a = find(true(i,:) == 1);
        b = find(true(i,:) ~=1 );
        for j=1:numel(a)
            for k=1:numel(b)
                if pred(i,a(j)) <= pred(i,b(k))
                    incor = incor +1;
                end
            end
        end
        temp = incor/(numel(a)*numel(b));
        loss = loss + temp;
    end
    loss = loss/m;

end

