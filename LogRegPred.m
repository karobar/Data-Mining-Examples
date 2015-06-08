function [ w ] = logRegPred( strippedX, y )
    % add a column of ones to the Xrix 
    X = [ones(size(strippedX, 1), 1) strippedX];

    % initial coefficients using ordinary least squares regression
%     size(X)
%     size(y)
%     X' * y
    w = pinv(X' * X) * X' * y;
    max_step = 100;     % maximum number of iterations
    tolerance = 0.01;   % threshold of change in weights (between iterations)
    eta = 0.1;
    step = 1;
    eps = tolerance;
    while step <= max_step & eps >= tolerance
        % vector of posterior probabilities that class equals 1
        p = logsig(X * w);  
        % store w vector, normalized
        w_old = w / sum(w);
        % calculate necessary matrices
        %P = diag(p .* (1 - p));
        vec = p .* (1 - p);
        n = length(vec);
        P = spdiags(vec(:),0,n,n);
        
        %E = diag(y - p);
        vec = y - p;
        n = length(vec);
        E = spdiags(vec(:),0,n,n);
        
        J = P * X;
        % apply full update rule for minimizing Euclidean distance
        %w = w + pinv(J' * J + J' * E * (2 * diag(p) - eye(size(X, 1))) * X) * (X' * P * (y - p));
        w = w + pinv(J' * J) * X' * P * (y - p); % Gauss-Newton approach
        %w = w + eta * X' * P * (y - p);
        
        % percent difference between old and new (normalized) w vectors
        eps = sum(abs(w_old - w / sum(w)));
        step = step + 1;
    end  
end

