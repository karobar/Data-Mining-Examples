%Travis Pressler

clear 
clc

fileNames = {'iris.data','sperm.data','haberman.data'};

for fileNum = 1:size(fileNames,2)
    currFile = fileNames{fileNum};
    currData = dlmread(currFile, ',');
    y = currData(:, size(currData, 2));
    strippedMAT = currData(:,1:size(currData, 2) - 1);
    
    for attribute = 1:size(strippedMAT,2)
        thinX = strippedMAT(:,attribute);
        
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

            currPerm = bigPerm(currStartPos:endPos);

            trainingSet = thinX;
            trainingYs = y;
            % separate training set and testing set.
            trainingSet(currPerm,:) = [];  
            trainingYs(currPerm,:) = [];

            means = [];
            stds  = [];
            maxes = [];
            mins  = []; 
            for col = 1:size(trainingSet,2)
                currCol = trainingSet(:,col);
                means = [means mean(currCol)];
                stds  = [stds  std(currCol)];
                maxes = [maxes max(currCol)];
                mins   = [mins  min(currCol)];
            end

            for row = 1:size(trainingSet,1)
                trainingSet(row,:) = minMax(trainingSet(row,:),mins,maxes);
            end

            %create model for Linear Regression
            LRweights = LogRegPred(trainingSet, trainingYs);

            %predict
            normedStrippedMAT = [];
            for row = 1:size(thinX,1)
                normedStrippedMAT(row,:) = minMax(thinX(row,:),mins,maxes);
            end
            truePredIndices = testLR(LRweights, currPerm, normedStrippedMAT);
            truePredKNNindices = KNNpred(trainingSet, trainingYs, currPerm', normedStrippedMAT);

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

        [~, ~, ~, LRAUC] = perfcurve(y, predictedClasses, 1);
        fprintf('%s Linear Regression AUC for attribute %d = %f\n', currFile, attribute, LRAUC);

        [~, ~, ~, KNNAUC] = perfcurve(y, predictedKNNclasses, 1);
        fprintf('%s K Nearest Neighbors AUC for attribute %d = %f\n', currFile, attribute, KNNAUC);
    end
end
