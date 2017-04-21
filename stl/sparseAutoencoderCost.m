function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
													lambda, sparsityParam, beta, data)
	W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
	W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), 		visibleSize, hiddenSize);
	b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize	+hiddenSize);
	b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

    m = size(data,2) ;
	a_1 = data ;
	
	z_2 = W1 * data + repmat(b1, 1, m) ;
	a_2 = sigmoid(z_2) ;
	rhocap = sum(a_2 , 2) / m ;
	
	z_3 = W2 * a_2  + repmat(b2, 1, m) ;
	a_3 = sigmoid(z_3)  ;

	w_ij = [ W1(:) ; W2(:) ] ;
	
	KL 		 = sparsityParam*log(sparsityParam./rhocap) + (1-sparsityParam)*log	((1-sparsityParam)./(1-rhocap)) ;
	sparsity = beta*(-1*(sparsityParam./rhocap) + (1 - sparsityParam)./(1-rhocap)) ;
	
	cost  = sum(sum(( a_3 - data ).^2))/(2*m) + (lambda/2)*(w_ij'*w_ij) + 	beta * sum(KL) ;
	
	delta_3 = -(data - a_3 ) .* fprime(z_3) ;
	delta_2 = ( W2' * delta_3 +  repmat(sparsity, 1, m)) .* fprime(z_2) ;
	
	W1grad = delta_2 * a_1'/m + lambda*W1 ;
	W2grad = delta_3 * a_2'/m + lambda*W2 ;
	b1grad = sum(delta_2, 2)/m  ;
	b2grad = sum(delta_3, 2)/m  ;
	grad   = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)] ;
end

function sigm = sigmoid(x)
  sigm = 1 ./ (1 + exp(-x));
end

function sigp = fprime(z)
	sigp = sigmoid(z).*(1-sigmoid(z)) ;
end
