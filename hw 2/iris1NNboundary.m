pkg load statistics

% setup data
D = csvread('iris.csv');
X_train = D(:, 1:2);
y_train = D(:, end); 

% setup meshgrid
[x1, x2] = meshgrid(2:0.01:5, 0:0.01:3);
grid_size = size(x1);
X12 = [x1(:) x2(:)];

kcount = [1,2,3,5,10,15]
kndx =1;
neighbors =[];
% compute kNN decision 
 n_X12 = size(X12, 1);
 decision = zeros(n_X12, 1);
 for entry = kcount
  for i=1:n_X12    
     point = X12(i, :);
    % compute euclidan distance from the point to all training data
     dist = pdist2(X_train, point);
     % sort the distance, get the index
     [~, idx_sorted] = sort(dist);
      % find the class of the nearest neighbour
      neighbors = idx_sorted;
      closest = y_train(neighbors(1:entry));
      dec_labels = unique(closest);
      count=[1:length(dec_labels)];
      
      for j=1 : length(dec_labels)
        for k= 1: length(closest)
           temp_label = dec_labels(j);
            if (closest(k) == temp_label)
                  count(j)= count(j)+1;
            endif
         endfor    
      endfor   
      
       temp = count(1);
       max_indx =1;
       for j= 2: length(count)
         if(count(j) > temp)
            temp = count(j);
            max_indx = j;
          endif 
        endfor  
        decision(i) = dec_labels(max_indx);
        a=mode(closest);
        b=decision(i);
        if a~= b
            a
            b
        endif    
     end
  % plot decisions in the grid
  
  decisionmap = reshape(decision, grid_size);
  figure, imagesc(2:0.01:5, 0:0.01:3, decisionmap);   % plot heading to give
  %(entry + "NN decisionmap");
  set(gca,'ydir','normal');

  % colormap for the classes
  % class 1 = light red, 2 = light green, 3 = light blue
  cmap = [1 0.8 0.8; 0.8 1 0.8; 0.8 0.8 1];
  colormap(cmap);

  % scatter plot data
  hold on;
  scatter(X_train(y_train == 1, 1), X_train(y_train == 1, 2), 10, 'r');
  scatter(X_train(y_train == 2, 1), X_train(y_train == 2, 2), 10, 'g');
  scatter(X_train(y_train == 3, 1), X_train(y_train == 3, 2), 10, 'b');
  hold off;
  end