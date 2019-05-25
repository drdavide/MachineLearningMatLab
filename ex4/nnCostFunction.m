function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% expand vector y in a matrix of single values
y_matrix = [1:num_labels] == y; % 5000x10

% perform forward propagation
a1 = [ones(m,1) X];             % 5000x401
z2 = a1 * Theta1';              % 5000x401 * 401x25 = 5000x25
a2 = sigmoid(z2);               % 5000x25
a2 = [ones(m,1), a2];           % 5000x26
z3 = a2 * Theta2';              % 5000x26 * 26x10 = 5000x10
a3 = sigmoid(z3);               % 5000x10

% cost function not regularized
%--------------------------------------------------------------------------------------
% Step 3 - Compute the unregularized cost according to ex4.pdf (top of Page 5), 
% (I had a hard time understanding this equation mainly that I had a misconception 
% that y(i)k is a vector, instead it is just simply one number) using a3, your ymatrix, 
% and m (the number of training examples). Cost should be a scalar value. 
% If you get a vector of cost values, you can sum that vector to get the cost.
% Remember to use element-wise multiplication with the log() function.

cost = sum(sum( y_matrix .* log(a3) + (1-y_matrix) .* log(1-a3) ));

J = -(1/m) * cost;

%----------------------------
% Step 4 - Compute the regularized component of the cost according to ex4.pdf 
% Page 6, using Θ1 and Θ2 (ignoring the columns of bias units), along with λ, and m. 
% The easiest method to do this is to compute the regularization terms separately, 
% then add them to the unregularized cost from Step 3.You can run ex4.m to check the 
% regularized cost, then you can submit Part 2 to the grader.
%----------------------------

reg = (lambda/(2*m)) * ( sum(sum( Theta1(:,2:end).^2 )) + sum(sum( Theta2(:,2:end).^2 )) );

J = J + reg;

% Sigmoid Gradient and Random Initialization
% Step 5 - You'll need to prepare the sigmoid gradient function g′(), as shown in 
% ex4.pdf Page 7

% SEE sigmoidGradient.m

% Step 6 - Implement the random initialization function as instructed on ex4.pdf, 
% top of Page 8. 

% SEE randInitializeWeights.m

% Backpropagation
% Step 7 - Now we work from the output layer back to the hidden layer, calculating how 
% bad the errors are. See ex4.pdf Page 9 for reference.δ3 equals the difference between 
% a3 and the y_matrix.δ2 equals the product of δ3 and Θ2 (ignoring the Θ2 bias units), 
% then multiplied element-wise by the g′() of z2 (computed back in Step 2).Note that at 
% this point, the instructions in ex4.pdf are specific to looping implementations, so the 
% notation there is different.Δ2 equals the product of δ3 and a2. This step calculates the 
% product and sum of the errors.Δ1 equals the product of δ2 and a1. This step calculates the 
% product and sum of the errors.

d3 = a3 - y_matrix;
d2_temp = (d3 * Theta2);
d2 = d2_temp(:,2:end) .* sigmoidGradient(z2);

Delta2 = d3' * a2;
Delta1 = d2' * a1;

% Gradient, non-regularized
% Step 8 - Now we calculate the non-regularized theta gradients, using the sums of the errors we 
% just computed. (see ex4.pdf bottom of Page 11)Θ1 gradient equals Δ1 scaled by 1/m Θ2 gradient 
% equals Δ2 scaled by 1/m 
% The ex4.m script will also perform gradient checking for you, using a smaller test case than the 
% full character classification example. So if you're debugging your nnCostFunction() using the 
% "keyboard" command during this, you'll suddenly be seeing some much smaller sizes of X and the 
% Θ values. Do not be alarmed. If the feedback provided to you by ex4.m for gradient checking 
% seems OK, you can now submit Part 4 to the grader. 


Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

% Gradient Regularization
% Step 9 - For reference see ex4.pdf, top of Page 12, for the right-most terms of the equation for j>=1.
% Now we calculate the regularization terms for the theta gradients. The goal is that regularization of 
% the gradient should not change the theta gradient(:,1) values (for the bias units) calculated in Step 8. 
% There are several ways to implement this (in Steps 9a and 9b).
% Method 1: 9a) Calculate the regularization for indexes (:,2:end), and 9b) add them to theta gradients (:,2:end).
% Method 2: 9a) Calculate the regularization for the entire theta gradient, then overwrite the (:,1) 
% value with 0 before 9b) adding to the entire matrix. 
% Details for Steps 9a and 9b:
% 9a) Pick a method, and calculate the regularization terms as follows:(λ/m)∗Θ1 (using either Method 1 or 
% Method 2)...and(λ/m)∗Θ2 (using either Method 1 or Method 2). 
% 9b) Add these regularization terms to the appropriate Θ1 gradient and Θ2 gradient terms from Step 8 
% (using either Method 1 or Method 2). Avoid modifying the bias unit of the theta gradients. 
% Note: there is an errata in the lecture video and slides regarding some missing parenthesis for this calculation. 
% The ex4.pdf file is correct. The ex4.m script will provide you feedback regarding the acceptable relative difference. 
% If all seems well, you can submit Part 5 to the grader.Now pat yourself on the back.

reg_Theta1 = (lambda/m) * Theta1;
reg_Theta2 = (lambda/m) * Theta2;

reg_Theta1(:,1) = 0;
reg_Theta2(:,1) = 0;

Theta1_grad = Theta1_grad + reg_Theta1;
Theta2_grad = Theta2_grad + reg_Theta2;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
