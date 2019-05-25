function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    h = X * theta;
    error_vector = h - y;
    %The change in theta (the "gradient") is the sum of the product of X and 
    %the "errors vector", scaled by alpha and 1/m. Since X is (m x n), and the 
    %error vector is (m x 1), and the result you want is the same size as theta 
    %(which is (n x 1), you need to transpose X before you can multiply it by the error vector.
    theta_change = alpha * (1/m) * (X' * error_vector); % The vector multiplication automatically includes calculating the sum of the products.
    
    % theta = theta - theta_change;
    theta = theta - theta_change;









    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
