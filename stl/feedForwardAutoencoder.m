function [activation] = feedForwardAutoencoder(theta, hiddenSize, visibleSize, data)
	W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
	b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
	numCases = size(data, 2) ;
	%activation = zeros(hiddenSize, numCases) ;
	%a_1 	   = data ;
	z_2        = W1 * data + repmat(b1, 1, numCases) ; 
	activation = sigmoid(z_2) ;
end

function sigm = sigmoid(x)
	sigm = 1 ./ (1 + exp(-x));
end
