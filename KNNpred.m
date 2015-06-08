function [ KNNpredictions ] = KNNpred( trainingSET, trainingYs, testSet, strippedMAT )
    testSetSize=size(testSet,1);
    trainingSetSize = size(trainingSET,1);
    K=5;
    KNNpredictions = [];
    for i = 1 : testSetSize
        u = strippedMAT(testSet(i),:);    
        for j = 1 : trainingSetSize
            d(j) = sqrt((trainingSET(j, :) - u) * (trainingSET(j, :) - u)'); % Euclidean distance
        end
        [~, b] = sort(d, 'ascend');
        yK = trainingYs(b(1 : K));
        n0 = length(find(yK == 0));
        n1 = K - n0;

        if n0 < n1
            KNNpredictions = [KNNpredictions ; testSet(i)]; 
        end
    end
end

