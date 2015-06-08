function [normed] = Znorm( unnormed , means, stds )
    for col = 1:size(unnormed,2)
        normed(:,col) = (unnormed(:,col) - means(:,col)) / stds(:,col);
    end
end

