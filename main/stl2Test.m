%% Evaluated on Test class
addpath ../minFunc/ 
addpath ../softmax/
addpath ../stl/

inputSize = 121 ;
numLabels = 2 ;
maxIter = 15 ;

trainData   = load('../dataset/myTrain2.txt') ;
testData    = load('../dataset/myTest2.txt') ;

trainSet = trainData ;
trainAttr = trainSet(:, 1:(end-1)) ;
trainLabel = trainSet(:, end) ;
maxTrain = max(trainSet(: , 1 : end-1)) ;
maxTrainInd = find( maxTrain > 1.0 ) ;
maxTrainRep = repmat(maxTrain, size(trainAttr,1),1) ;
trainAttr(:, maxTrainInd) = trainAttr(:, maxTrainInd)./maxTrainRep(:, maxTrainInd) ;
trainAttr = trainAttr' ;
trainLabel= trainLabel' ;
clear trainSet ;

testSet = testData ;
testAttr = testSet(:, 1:(end-1)) ;
testLabel = testSet(:, end) ;
maxTest = max(testSet(: , 1 : end-1)) ;
maxTestInd = find( maxTest > 1.0 ) ;
maxTestRep = repmat(maxTest, size(testAttr,1),1) ;
testAttr(:, maxTestInd)  = testAttr(:, maxTestInd)./maxTestRep(:, maxTestInd) ;
testAttr = testAttr' ;
testLabel = testLabel' ;
clear testSet ;
    
hs = 30 ;
beta = 3 ;
sp = .05 ;
l1 = 0.00001 ;
theta = csvread('../dataset/theta_88_392.csv');
options.Method = 'lbfgs' ; 
options.maxIter = maxIter ;	
options.useMex = 0  	  ;
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, inputSize, ...
				hs, l1, sp, beta, trainAttr), theta, options) ;
	  
trainFeatures = feedForwardAutoencoder(opttheta, hs, inputSize, ...
                                       trainAttr) ;
testFeatures = feedForwardAutoencoder(opttheta, hs, inputSize, ...
                                       testAttr)  ;
options.Method = 'lbfgs' ; 
lambda = 0.1 ;
options.Method = 'lbfgs';
options.useMex = 0      ;
options.maxIter = 500   ;
inputSize = size(trainFeatures, 1) ;
numClasses = 2 ;
inputData = trainFeatures ;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
 	                       inputData, trainLabel, options) ;
[pred] = softmaxPredict(softmaxModel, testFeatures) ;
acc = mean(testLabel(:) == pred(:)) ;
[fmeas, prec, recall] = prcrec(testLabel(:), pred(:), 2) ;
prc = prec(:) ;
rec = recall(:) ;
fm = fmeas(:) ;
avgAcc = acc * 100 ;

fprintf('Accuracy: %0.3f\n', avgAcc) ;
fprintf('Precision: %0.3f\n', prc) ;
fprintf('Recall: %0.3f\n', rec) ;
fprintf('F-measure: %0.3f\n', fmeas) ;
fprintf('\n');
