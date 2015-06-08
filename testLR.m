function [ TruePredIndices ] = testLR( weights, testSetIndices, strippedMAT )
    TruePredIndices = [];
    for i = 1:size(testSetIndices,2)
        currEntry = strippedMAT(testSetIndices(i),:);
        currEntry = [1 currEntry];
        %weights
        %currEntry
        output = dot(weights',currEntry');
        
        if (output) > 0
            TruePredIndices = [TruePredIndices ; testSetIndices(i)];
        end
    end
    
    %TruePredIndices
end

