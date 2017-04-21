function[fmeas, prec, recall] = prcrec(actual, pred, level)
	prec = zeros(1, level-1) ;
	recall = zeros(1, level-1) ;
    fmeas = zeros(1, level-1) ;
	for i=1:level-1
		ab  = actual==(i+1) ;
		pb  = pred==(i+1)  ;
		ind = pb == 1 ;
		tp  = sum(ab(ind)) ;
		pp  = sum(pb) ;
		ind  = ab==1  ; 
		ap   = sum(ab(ind)) ;
		prec(i) = tp/pp ;
		recall(i) = tp/ap ;
		fmeas(i) = (2 * prec(i) * recall(i) )/(prec(i) + recall(i)) ;
		prec(i) = tp/pp * 100 ;
		recall(i) = tp/ap *100 ;
	end
end
 

