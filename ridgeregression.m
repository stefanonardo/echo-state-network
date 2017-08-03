function [ W ] = ridgeregression( X, Y, esn)

W = Y'*X'*inv(X*X'+esn.lambda*eye(size(X,1))); 

end

