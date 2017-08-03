function [ W ] = pseudoinverse( X, Y, esn)

W = Y' * pinv(X);

end

