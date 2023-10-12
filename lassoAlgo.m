function [optimal_lambda, optimal_coefficients] = lassoAlgo(A,y)
% Define lambda range 
lambda_vals = [0.01 0.1 1 10 50 1000];

% Define number of folds for cross-validation
num_folds = 3;

% Initialize arrays to store validation errors and coefficients
validation_errors = zeros(length(lambda_vals),1);
coefficients = zeros(length(lambda_vals),size(A,2));
% Perform cross-validation
for i = 1:length(lambda_vals)
    lambda = lambda_vals(i);
    for j = 1:num_folds
        % Split data into training and validation sets
        idx = crossvalind('Kfold', size(A,1), num_folds);
        train_A = A(idx~=j,:);
        train_b = y(idx~=j);
        valid_A = A(idx==j,:);
        valid_b = y(idx==j);
        
        % Define optimization variables and objective function
        x = sdpvar(size(A,2),1,'full','complex');
        obj = norm(train_A*x - train_b)^2 + lambda*norm(x,1);
        
        % Solve optimization problem using fmincon solver
        constraints = [];
        options = sdpsettings('solver','sdpt3','verbose',0,'cachesolvers',1);
        sol = optimize(constraints,obj,options);
        
        % Compute validation error and store coefficients
        if sol.problem == 0
            validation_errors(i) = validation_errors(i) + norm(valid_A*value(x) - valid_b)^2;
            coefficients(i,:) = coefficients(i,:) + transpose(value(x));
        end
    end
end

% Compute mean validation error and coefficients over folds
validation_errors = validation_errors./num_folds;
coefficients = coefficients./num_folds;

% Find lambda value that minimizes validation error
[~,min_idx] = min(validation_errors);
optimal_lambda = lambda_vals(min_idx);
optimal_coefficients = coefficients(min_idx,:);

