function data = ipOversampling(data, weights, normalize)
    if size(data, 1) ~= length(weights)
        error('Length of weights and observations in data must coincide.');
    end
    
    if ~isnumeric(weights)
        error('Weights are required to be numeric.');
    end
    
    if min(weights) <= 0
        error('All weights are required to be greater than 0.');
    end
    
    if normalize
        weights = weights / min(weights); % let minimum weight equal one --> as less observations as possible (as much as necessary)
    end
    
    pos = repelem(1:size(data, 1), round(weights));
    data = data(pos, :);
end
