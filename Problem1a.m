%Travis Pressler

clear 
clc

fileNames = {'iris.data','sperm.data','haberman.data'};

for fileNum = 1:size(fileNames,2)
    currFile = fileNames{fileNum};
    currData = dlmread(currFile, ',');
    y = currData(:, size(currData, 2));
    strippedMAT = currData(:,1:size(currData, 2) - 1);

    K = 10;
    N = size(strippedMAT, 1);
    testSetSize = idivide(N,int16(K)) + rem(N, K);
    predictedClasses = zeros(N,1);
    predictedKNNclasses = zeros(N,1);

    bigPerm = randperm(N);
    %size(bigPerm);
    currStartPos = 1;
    for foldNum = 1:K 
        endPos = currStartPos+testSetSize-1;

        currPerm = bigPerm(currStartPos:endPos-1);

        trainingSet = strippedMAT;
        trainingYs = y;
        % separate training set and testing set.
        trainingSet(currPerm,:) = [];  
        trainingYs(currPerm,:) = [];  

        %LRweights = mnrfit(trainingSet, trainingYs
        LRweights = LogRegPred(trainingSet, trainingYs);
        %currPerm
        truePredIndices = testLR(LRweights, currPerm, strippedMAT);
        truePredKNNindices = KNNpred(trainingSet, trainingYs, currPerm', strippedMAT);
        predictedClasses(truePredIndices, :) = 1;
        predictedKNNclasses(truePredKNNindices, :) = 1;

        currStartPos = currStartPos + testSetSize;
        testSetSize = idivide(N,int16(K));  
    end

    LRsuccesses = 0;
    KNNsuccesses = 0;
    for currClass = 1: size(predictedClasses,1);
        if predictedClasses(currClass) == y(currClass)
            LRsuccesses = LRsuccesses + 1;
        end
        if predictedKNNclasses(currClass) == y(currClass)
            KNNsuccesses = KNNsuccesses + 1;
        end
    end
    
    %[y predictedClasses predictedKNNclasses]

    [X, Y, ~, LRAUC] = perfcurve(y, predictedClasses, 1);
    figure();
    plot(X,Y);
    title(strcat(currFile , '-Linear Regression'));
    xlabel('False positive rate'); ylabel('True positive rate')
    LRaccuracy = LRsuccesses / size(predictedClasses,1)
    fprintf('%s linear regression accuracy = %f\n', currFile, LRaccuracy);
    fprintf('%s AUC = %f\n', currFile, LRAUC);

    [X, Y, ~, KNNAUC] = perfcurve(y, predictedKNNclasses, 1);
    figure();
    plot(X,Y);
    title(strcat(currFile, '-K Nearest Neighbors'));
    xlabel('False positive rate'); ylabel('True positive rate')
    KNNaccuracy = KNNsuccesses / size(predictedClasses,1)
    fprintf('%s K nearest neighbors accuracy = %f\n', currFile, KNNaccuracy);
    fprintf('%s AUC = %f\n', currFile, KNNAUC);
end
