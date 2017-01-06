% printing option
more off;

% read files
D_tr = csvread('spambasetrain.csv'); 
D_ts = csvread('spambasetest.csv');  

% construct x and y for training and testing
X_tr = D_tr(:, 1:end-1);
y_tr = D_tr(:, end);
X_ts = D_ts(:, 1:end-1);
y_ts = D_ts(:, end);

% number of training / testing samples
n_tr = size(D_tr, 1);
n_ts = size(D_ts, 1);

% add 1 as a feature
X_tr = [ones(n_tr, 1) X_tr];
X_ts = [ones(n_ts, 1) X_ts];

% perform gradient descent :: logistic regression
n_vars = size(X_tr, 2);              % number of variables
lr = 1e-3;                           % learning rate
w = zeros(n_vars, 1);                % initialize parameter w
tolerance = 1e-2;                    % tolerance for stopping criteria
lambda = [0.00390625,0.015625,0.0625,0.25,1,4];
accuracy_tr = size(length(lambda),1);
accuracy_ts = size(length(lambda),1);
for k = 1:length(lambda)
  iter = 0;                            % iteration counter
  max_iter = 1000;                     % maximum iteration
  while true
      iter = iter + 1;                 % start iteration
      % calculate gradient
      grad = zeros(n_vars, 1);         % initialize gradient
      for j=1:n_vars
        temp = (X_tr'(j,:))';
        temp_exp = exp(w(j) * temp);
        temp_lambda = (lambda(k) * w(j));
        temp_grad = (y_tr .* temp) - ((temp .* temp_exp)./(1 .+ temp_exp));
        grad(j) = sum(temp_grad')- temp_lambda;
        %grad(j) =             % compute the gradient with respect to w_j here
      end

      % take step
      % w_new = w + .....              % take a step using the learning rate
      w_new = w .+ (lr * grad);

    
     % printf('lambda=%d ,iter = %d, mean abs gradient = %0.3f\n',lambda(k), iter, mean(abs(grad)));
      %fflush(stdout);

      %  stopping criteria and perform update if not stopping
      if mean(abs(grad)) < tolerance
          %w = w_new;
          break;
      else
          w = w_new;
      end

      if iter >= max_iter 
          break;
      end
  end

  % use w for prediction
  pred = zeros(n_ts, 1);               % initialize prediction vector

  for i=1:n_ts
    temp_sum = sum(w' .* X_ts(i,:));
    prob_0 = 1 / (1+ exp(temp_sum));
    prob_1 = exp(temp_sum)/(1+ exp(temp_sum));
    % pred(i) = .....                % compute your prediction
    if(prob_1 >= prob_0)
      pred(i) = 1;          %ham
    else
      pred(i) = 0;          %spam 
    endif  
  end

  % calculate testing accuracy
  count =0;
  for i= 1:n_ts
    if(pred(i)== y_ts(i))
      count++;
    end
  end
  accuracy_ts(k) = count / n_ts;  
  printf('\nlambda=%d, Testing accuracy=%d',lambda(k),accuracy_ts(k));

  % repeat the similar prediction procedure to get training accuracy
  pred_tr = zeros(n_tr, 1);               % initialize prediction vector

  for i=1:n_tr
      temp_sum = sum(w' .* X_tr(i,:));
      prob_0 = 1 / (1+ exp(temp_sum));
      prob_1 = exp(temp_sum)/(1+ exp(temp_sum));
      % pred(i) = .....                % compute your prediction
      if(prob_1 >= prob_0)
        pred_tr(i) = 1;          %ham
      else
        pred_tr(i) = 0;          %spam 
      endif  
  end

  % calculate testing accuracy
  count =0;
  for i= 1:n_tr
    if(pred_tr(i)== y_tr(i))
      count++;
    end
  end
  accuracy_tr(k) = count / n_tr; 
  printf('\nlambda=%d, Training accuracy=%d',lambda(k),accuracy_tr(k));
end

k=[-8,-6,-4,-2,0,2];
plot(k,accuracy_tr,"-;trainaccuracy;",k,accuracy_ts,"-r;testaccuracy;");
xlabel('Power of 2 for lambda');
ylabel('Accuracy');  
