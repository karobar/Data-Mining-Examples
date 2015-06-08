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

    
    %testSetSize = idivide(N,int16(K)) + rem(N, K);
    %predictedClasses = zeros(N,1);
    %predictedKNNclasses = zeros(N,1);
    
    N = size(X, 1);
    testSetSize = idivide(N,int16(K)) + rem(N, K);
    testSet = [];
    
    bigPerm = randperm(size(X,1));
    currStartPos = 1;
    NNpreds = zeros(N,1);
    DTpreds = zeros(N,1);
    for foldNum = 1:K 
        %foldNum
        endPos = currStartPos+testSetSize-1;

        currPerm = bigPerm(currStartPos:endPos);
        
        trainingSet = X;
        trainingSet(currPerm,:) = [];
        trainingYs = y;
        trainingYs(currPerm,:) = [];
        
        testSet = X(currPerm,:);
        
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
        
        %predictions
        for currTest = 1:size(testSet,1)
            p = 0;
            for b = 1 : length(net)
                p = p + sim(net{b}, X(currPerm(currTest),:)');
            end
            p = p ./ length(net);
            if p > 0.5
                NNpreds(currPerm(currTest),1) = 1;
            end
            
            p = 0;
            for b = 1 : length(t)
                p = p + treeval(t{b}, X(currPerm(currTest),:))';
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

    [X, Y, ~, NNAUC] = perfcurve(y, NNpreds, 1);
    figure();
    plot(X,Y);
    title(strcat(currFile , '-Neural Network'));
    xlabel('False positive rate'); ylabel('True positive rate')
    NNaccuracy = NNsuccesses / size(NNpreds,1)
    fprintf('%s Neural Network accuracy = %f\n', currFile, NNaccuracy);
    fprintf('%s Neural Network AUC = %f\n', currFile, NNAUC);

    [X, Y, ~, DTAUC] = perfcurve(y, DTpreds, 1);
    figure();
    plot(X,Y);
    title(strcat(currFile, '-Decision Trees'));
    xlabel('False positive rate'); ylabel('True positive rate')
    DTaccuracy = DTsuccesses / size(DTpreds,1)
    fprintf('%s Decision Trees accuracy = %f\n', currFile, DTaccuracy);
    fprintf('%s Decision Trees AUC = %f\n', currFile, DTAUC);
end
