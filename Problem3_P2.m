%Travis Pressler
clear 
clc
% 'EEG Eye State.data'
fileNames = {'iris.data','sperm.data', 'haberman.data'};

for fileNum = 1:size(fileNames,2)
    currFile = fileNames{fileNum};
    currData = dlmread(currFile, ',');
    
    % neural network parameters
    h = 5;   % hidden neurons
    B = 10;  % number of bagged models
    K = 10;  % number of folds
    trees = 100; 
    
    y = currData(:, size(currData, 2));
    X = currData(:,1:size(currData, 2) - 1);
    for attribute = 1:size(X,2)    
        thinX = X(:,attribute);
        
        N = size(thinX, 1);
        testSetSize = idivide(N,int16(K)) + rem(N, K);
        testSet = [];

        bigPerm = randperm(size(thinX,1));
        currStartPos = 1;
        NNpreds = zeros(N,1);
        DTpreds = zeros(N,1);
        for foldNum = 1:K 
            %foldNum
            endPos = currStartPos+testSetSize-1;

            currPerm = bigPerm(currStartPos:endPos);

            trainingSet = thinX;
            trainingSet(currPerm,:) = [];
            trainingYs = y;
            trainingYs(currPerm,:) = [];

            testSet = thinX(currPerm,:);

            %%DATA NORMALIZATION
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
            %%DATA NORMALIZATION

            %generate 10 neural networks
            for b = 1 : B      
                % sample with replacement
                q = ceil(rand(1, size(trainingSet, 1)) * size(trainingSet, 1));     
                % form a bootstrapped training set
                Xb = trainingSet(q, :);
                yb = trainingYs(q, 1);

                % initialize and train b-th neural network
                net{b} = newff(Xb', yb', h, {'tansig', 'tansig'}, 'trainrp');
                net{b}.trainParam.epochs = 100;
                net{b}.trainParam.show = NaN;
                net{b}.trainParam.showWindow = false;
                net{b}.trainParam.max_fail = 10;
                net{b}.divideFcn = 'divideind';
                net{b}.divideParam.trainInd = 1 : size(Xb, 1);
                net{b}.divideParam.valInd = [];
                net{b}.divideParam.testInd = [];
                net{b} = train(net{b}, Xb', yb');
            end

            % generate 100 decision trees in cell array t
            for b = 1 : trees
                % sample with replacement
                q = ceil(rand(1, size(trainingSet, 1)) * size(trainingSet, 1));     
                % form a bootstrapped training set
                Xb = trainingSet(q, :);
                yb = trainingYs(q, 1);

                t{b} = treefit(Xb, yb, 'splitmin', 3);
            end       

            normedX = [];
            for row = 1:size(thinX,1)
                normedX(row,:) = Znorm(thinX(row,:),mins,maxes);
            end

            %predictions
            for currTest = 1:size(testSet,1)
                p = 0;
                for b = 1 : length(net)
                    p = p + sim(net{b}, normedX(currPerm(currTest),:)');
                end
                p = p ./ length(net);
                if p > 0.5
                    NNpreds(currPerm(currTest),1) = 1;
                end

                p = 0;
                for b = 1 : length(t)
                    p = p + treeval(t{b}, normedX(currPerm(currTest),:))';
                end
                p = p ./ length(t);
                if p > 0.5
                    DTpreds(currPerm(currTest),1) = 1;
                end
            end

            currStartPos = currStartPos + testSetSize;
            testSetSize = idivide(N,int16(K));      
        end
        %[y NNpreds DTpreds]

        NNsuccesses = 0;
        DTsuccesses = 0;
        for currClass = 1: size(NNpreds,1);
            if NNpreds(currClass) == y(currClass)
                NNsuccesses = NNsuccesses + 1;
            end

            if DTpreds(currClass) == y(currClass)
                DTsuccesses = DTsuccesses + 1;
            end
        end

        
        [~, ~, ~, NNAUC] = perfcurve(y, NNpreds, 1);
        fprintf('%s Neural Network AUC for attribute %d = %f\n', currFile, attribute, NNAUC);

        [~, ~, ~, DTAUC] = perfcurve(y, DTpreds, 1);
        fprintf('%s Decision Tree AUC for attribute %d = %f\n', currFile, attribute, DTAUC);
    end
end
