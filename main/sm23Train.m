%% softmax regression code for anomaly detection
addpath ../softmax/
addpath ../minFunc/
inputSize = 121 ;
iter = 10 ;
data = load('../dataset/myTrain23.txt') ;
lamArr = [0, 0.00005, 0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01] ;
for l=1:size(lamArr,2)
fprintf('Computing for lambda : %0.5f\n',lamArr(l));
X = [1, 12598, 25195, 37792, 50389, 62986, 75583, 88180, 100777, 113374, 125973] ;
resArr = zeros(1, iter) ;
for i=1:iter
	low =  X(i) ;
	up = X(i+1)-1 ;
	testInd =  low:up ;
	trainInd =  [X(1):X(i)-1, up+1:X(11)] ;
	trainSet = data(trainInd, :) ;
	trainAttr = trainSet(:, 1:(end-1)) ;	
	trainLabel = trainSet(:, end) ; 	
	maxTrain = max(trainSet(: , 1 : end-1)) ;
	maxTrainInd = find( maxTrain > 1.0 ) ;
    maxTrainRep = repmat(maxTrain, size(trainAttr,1),1) ;
    trainAttr(:, maxTrainInd) = trainAttr(:, maxTrainInd)./maxTrainRep(:, maxTrainInd) ;
	trainAttr = trainAttr'  ;
	trainLabel = trainLabel' ;
	clear trainSet ;
	
	testSet = data(testInd, :) ;
	testAttr = testSet(:, 1:(end-1)) ;
	testLabel = testSet(:, end) ;
	maxTest = max(testSet(: , 1 : end-1)) ;
    maxTestInd = find( maxTest > 1.0 ) ;
    maxTestRep = repmat(maxTest, size(testAttr,1),1) ;
    testAttr(:, maxTestInd)  = testAttr(:, maxTestInd)./maxTestRep(:, maxTestInd) ;
	testAttr = testAttr'  ;
	testLabel = testLabel' ;
	clear testSet ;

	lambda = lamArr(l) ;
	options.Method = 'lbfgs' ; 
	options.useMex = 0 ;  
	options.maxIter = 1000 ;
	inputSize = 121 ;
	numClasses = 23 ;
	inputData = trainAttr ;
	softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                         inputData, trainLabel, options) ;
	[pred] = softmaxPredict(softmaxModel, testAttr) ;
	acc = mean(testLabel(:) == pred(:)) * 100 ;
	resArr(i) = acc ;
end
fprintf('\n******* Accuracy calculation ******\n') ;
for j=1:iter
	fprintf('%0.2f\n', resArr(j)) ;
end
end
