%% softmax regression code for anomaly detection
addpath ../softmax/
addpath ../minFunc/

avgAcc = zeros(1, 10) ;
prc_vec = zeros(1, 10) ;
rec_vec = zeros(1, 10) ;
fmeas_vec = zeros(1, 10) ;

trainData = load('../dataset/myTrain2.txt') ;
testData = load('../dataset/myTest2.txt') ;

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
	
options.Method = 'lbfgs' ; 
lambda = 0.05 ;
options.Method = 'lbfgs';
options.useMex = 0 ;
options.maxIter = 1000 ;
inputSize = size(trainAttr, 1) ;
numClasses = 2 ;
inputData = trainAttr ;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
  	                       inputData, trainLabel, options) ;
[pred] = softmaxPredict(softmaxModel, testAttr) ;
acc = mean(testLabel(:) == pred(:)) ;
[fmeas, prec, recall] = prcrec(testLabel(:), pred(:), 2) ;
prc_vec(:,1) = prec(:) ;
rec_vec(:,1) = recall(:) ;
fmeas_vec(:,1) = fmeas(:) ;
avgAcc(1) = acc*100 ;

fprintf('Accuracy: %0.3f\n', avgAcc(1)) ;
fprintf('Precision: %0.3f\n',prc_vec(1)) ;
fprintf('Recall: %0.3f\n', rec_vec(1)) ;
fprintf('F-meausre: %0.3f\n', fmeas_vec(1)) ;
