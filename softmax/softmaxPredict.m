function [pred] = softmaxPredict(softmaxModel, data)
	theta = softmaxModel.optTheta; 
	M = theta * data ;
	M = bsxfun(@minus, M, max(M, [], 1));
	expM = exp(M) ;
	h = bsxfun(@rdivide, expM, sum(expM)) ;
	[~, ind] = max(h, [], 1) ;
	pred = ind ;
end

