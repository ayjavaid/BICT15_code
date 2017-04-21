function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)
	theta = reshape(theta, numClasses, inputSize) ;
	numCases = size(data, 2) ;
	groundTruth = full(sparse(labels, 1:numCases, 1));
	M = theta * data ;
	M = bsxfun(@minus, M, max(M, [], 1));
	expM = exp(M) ;
	h = bsxfun(@rdivide, expM, sum(expM)) ;
	decay = (lambda * theta(:)'*theta(:))/2 ;
	cost = (-1*sum(sum((groundTruth .* log(h)))))/numCases + decay ;
	thetagrad = -1*((groundTruth - h) * data')/numCases + lambda * theta ;  
	grad = thetagrad(:);
end

