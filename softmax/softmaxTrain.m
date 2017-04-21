function [softmaxModel] = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options)
	if ~exist('options', 'var')
		options = struct;
	end

	if ~isfield(options, 'maxIter')
		options.maxIter = 400;
	end
	theta = 0.005 * randn(numClasses * inputSize, 1);
	%addpath minFunc/
	options.Method = 'lbfgs'; 
    %minFuncOptions.display = 'on';
	[softmaxOptTheta, ~] = minFunc( @(p) softmaxCost(p, ...
                                   numClasses, inputSize, lambda, ...
                                   inputData, labels), ...                                   
                              theta, options);
	softmaxModel.optTheta = reshape(softmaxOptTheta, numClasses, inputSize);
	softmaxModel.inputSize = inputSize;
	softmaxModel.numClasses = numClasses;        
end                          
