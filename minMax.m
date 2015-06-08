function [normed] = minmax( unnormed , min, max )
    for col = 1:size(unnormed,2)
        normed(:,col) = (unnormed(:,col) - min(:,col)) / (max(:,col)-min(:,col));
    end
end

