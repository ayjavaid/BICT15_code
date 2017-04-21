addpath ../minFunc/ 
addpath ../softmax/
addpath ../stl/

numLabels = 5 ;
maxIter = 500 ;
prc_vec = zeros(4, 1) ;
rec_vec = zeros(4, 1) ;
fmeas_vec = zeros(4, 1) ;

trainData = load('../dataset/myTrain5.txt') ;
testData = load('../dataset/myTest5.txt')	;
trainSet = trainData ;
trainAttr = trainSet(:, 1:(end-1)) ;
trainLabel = trainSet(:, end) ;
maxTrain = max(trainSet(: , 1 : end-1)) ;
maxTrainInd = find( maxTrain > 1.0 ) ;
maxTrainRep = repmat(maxTrain, size(trainAttr,1),1) ;
trainAttr(:, maxTrainInd) = trainAttr(:, maxTrainInd)./maxTrainRep(:, maxTrainInd) ;
trainAttr = trainAttr' ;
trainLabel = trainLabel' ;
clear trainSet ;

testSet = testData ;
testAttr = testSet(:, 1:(end-1)) ;
testLabel = testSet(:, end) ;
maxTest = max(testSet(: , 1 : end-1)) ;
maxTestInd = find( maxTest > 1.0 ) ;
maxTestRep = repmat(maxTest, size(testAttr,1),1) ;
testAttr(:, maxTestInd) = testAttr(:, maxTestInd)./maxTestRep(:, maxTestInd) ;
testAttr = testAttr' ;
testLabel = testLabel' ;
clear testSet ;

hs = 60 ;
beta = 5 ;
sp = .25 ;
l1 = 0.00001 ;
inputSize = 121 ;
theta=csvread('../dataset/init5.csv');
options.Method = 'lbfgs' ; 
options.maxIter = maxIter ;	
options.useMex = 0 ;
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, inputSize, ...
				hs, l1, sp, beta, trainAttr), theta, options) ;
	  
trainFeatures = feedForwardAutoencoder(opttheta, hs, inputSize, ...
                                       trainAttr) ;
testFeatures = feedForwardAutoencoder(opttheta, hs, inputSize, ...
                                       testAttr)  ;
options.Method = 'lbfgs' ; 
lambda = 0 ;
options.Method = 'lbfgs' ;
options.useMex = 0 ;
options.maxIter = 500 ; 
inputSize = size(trainFeatures, 1) ;
numClasses = 5 ;
inputData = trainFeatures ;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
  	                       inputData, trainLabel, options) ;
[pred] = softmaxPredict(softmaxModel, testFeatures) ;
acc = mean(testLabel(:) == pred(:)) ;
[fmeas, prec, recall] = prcrec(testLabel(:), pred(:), 5) ;
prc_vec(:,1) = prec(:) ;
rec_vec(:,1) = recall(:) ;
fmeas_vec(:,1) = fmeas(:) ;
avgAcc = acc*100 ;

fprintf('Accuracy: %0.3f\n', avgAcc) ;
fprintf('******Precision calculation*****\n') ;
for j=1:4
		fprintf('%0.3f ', prc_vec(j,1)) ;
end
fprintf('\n') ;

fprintf('******Recall calculation*****\n') ;
for j=1:4
		fprintf('%0.3f ', rec_vec(j,1)) ;
end
fprintf('\n') ;

fprintf('******F-measure calculation*****\n') ;
for j=1:4	
    fprintf('%0.3f ', fmeas_vec(j,1)) ;
end
fprintf('\n') ;
