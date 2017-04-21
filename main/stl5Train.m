%% softmax regression code for anomaly detection
addpath ../softmax/
addpath ../minFunc/
addpath ../stl
maxIter = 1000 ; 
inputSize = 121 ;
iter = 10 ;
prc_vec = zeros(4, 10) ;
rec_vec = zeros(4, 10) ;
fmeas_vec = zeros(4, 10) ;

data = load('../dataset/myTrain5.txt') ;
X = [1, 12598, 25195, 37792, 50389, 62986, 75583, 88180, 100777, 113374, 125973] ;
resArr = zeros(1, iter) ;
for i=1:iter
	low =  X(i);
	up = X(i+1)-1 ;
	testInd = low:up ;
	trainInd = [X(1):X(i)-1, up+1:X(11)] ;
	trainSet = data(trainInd, :) ;
	trainAttr = trainSet(:, 1:(end-1)) ;	
	trainLabel = trainSet(:, end) ; 	
	maxTrain = max(trainSet(: , 1 : end-1)) ;
    maxTrainInd = find( maxTrain > 1.0 ) ;
    maxTrainRep = repmat(maxTrain, size(trainAttr,1),1) ;
    trainAttr(:, maxTrainInd) = trainAttr(:, maxTrainInd)./maxTrainRep(:, maxTrainInd) ;
	trainAttr = trainAttr' ;
	trainLabel = trainLabel' ;
	clear trainSet ;
	
	testSet = data(testInd, :)	;
	testAttr = testSet(:, 1:(end-1)) ;
	testLabel = testSet(:, end) ;
	maxTest = max(testSet(: , 1 : end-1)) ;
    maxTestInd = find( maxTest > 1.0 ) ;
    maxTestRep = repmat(maxTest, size(testAttr,1),1) ;
    testAttr(:, maxTestInd)  = testAttr(:, maxTestInd)./maxTestRep(:, maxTestInd) ;
	testAttr = testAttr' ;
	testLabel = testLabel' ;
	clear testSet ;

	hs = 110 ;
	beta=4 ;
	sp = 0.15 ;
	l1 = 0.00005 ;
	inputSize = 121 ;
	theta = initializeParameters(hs, inputSize) ;
	options.Method = 'lbfgs' ; 
	options.maxIter = maxIter ;	
	options.useMex = 0 ;
	[opttheta, cost]= minFunc( @(p) sparseAutoencoderCost(p, inputSize, ...
					hs, l1, sp, beta, trainAttr), theta, options) ;
		  
	trainFeatures = feedForwardAutoencoder(opttheta, hs, inputSize, ...
	                                       trainAttr) ;
	testFeatures = feedForwardAutoencoder(opttheta, hs, inputSize, ...
	                                       testAttr)  ;

	lambda = 0.00001 ;
	options.Method  = 'lbfgs' ; 
	options.useMex  = 0 ;  
	options.maxIter = maxIter ;
	inputSize = size(trainFeatures, 1) ;
	numClasses = 5 ;
	inputData = trainFeatures ;
	softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                        inputData, trainLabel, options) ;
	[pred] = softmaxPredict(softmaxModel, testFeatures) ;
	acc = mean(testLabel(:) == pred(:)) * 100 ;
	resArr(i) = acc ;
	[fmeas, prec, recall] = prcrec(testLabel(:), pred(:), 5) ;
	prc_vec(:,i) = prec(:) ;
	rec_vec(:,i) = recall(:) ;
	fmeas_vec(:,i) = fmeas(:) ;
end

fprintf('******* Accuracy calculation ******\n') ;
for i=1:iter
	fprintf('%0.2f\n', resArr(i)) ;
end

fprintf('******Precision calculation*****\n') ;
for i=1:10
	fprintf('%0.2f, %0.2f, %0.2f, %0.2f\n', prc_vec(1,i), prc_vec(2,i), prc_vec(3,i), prc_vec(4,i)) ;
end

fprintf('******Recall calculation*****\n') ;
for i=1:10
	fprintf('%0.2f, %0.2f, %0.2f, %0.2f\n', rec_vec(1,i), rec_vec(2,i), rec_vec(3,i), rec_vec(4,i)) ;
end

fprintf('******F-measure calculation*****\n') ;
for i=1:10
	fprintf('%0.2f, %0.2f, %0.2f, %0.2f\n', fmeas_vec(1,i), fmeas_vec(2,i), fmeas_vec(3,i), fmeas_vec(4,i)) ;
end

