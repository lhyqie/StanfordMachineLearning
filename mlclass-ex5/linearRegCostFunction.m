function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


for i = 1: m 

          J =  J + ( X(i,:) * theta - y(i))^2;	
  
end

J = J / (2 * m);


R = 0;
n = length(theta);

for j = 2:n
         R = R + theta(j)^2; 
end

R = R * lambda / (2 * m);

J = J + R;

% --------------------------------------

grad =  X' * ( X * theta - y ) /m;
grad2 = (lambda/m).* theta;
grad2(1) = 0;
grad = grad + grad2;

% =========================================================================

grad = grad(:);

end
