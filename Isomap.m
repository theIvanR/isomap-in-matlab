function [Y, R, E] = Isomap(D, n_fcn, n_size, optsIn)
% ISOMAP   Performs an Isomap embedding with optional GPU acceleration.
%
%   [Y, R, E] = Isomap(D, n_fcn, n_size, optsIn)
%
%   Input:
%     D       - N x N distance matrix (must be symmetric and nonnegative).
%     n_fcn   - Neighborhood function: 'epsilon' or 'k' (case-insensitive).
%     n_size  - Neighborhood size (if 'epsilon', a distance threshold; if 'k', 
%               the number of neighbors).
%
%     optsIn  - (Optional) structure with fields:
%                dims        : Vector of embedding dimensions to try (default: 1:10)
%                comp        : Which connected component to embed (default: 1, largest)
%                overlay     : Flag to overlay the neighborhood graph on 2D embedding (default: 0)
%                verbose     : Flag to display progress messages (default: 1)
%                computeMode : 'gpu' or 'cpu' to select computation mode (default: 'gpu')
%
%   Output:
%     Y.coords - Cell array of embedded coordinates (each column is a point) for 
%                each dimension specified in optsIn.dims.
%     Y.index  - Indices of the points in the selected connected component.
%     R        - Residual variances for each embedding (lower is better).
%     E        - Edge matrix for the neighborhood graph (if overlay==1; empty otherwise).
%
%   Example:
%      [Y, R, E] = Isomap(D, 'k', 10, struct('dims',1:5, 'verbose',1, 'computeMode','gpu'));

%% 0: INPUT VALIDATION & DEFAULTS
    if nargin < 3
        error('Isomap requires at least D, n_fcn, and n_size.');
    end
    if size(D,1) ~= size(D,2)
        error('Distance matrix D must be square.');
    end
    % Neighborhood function
    if strcmpi(n_fcn, 'k')
        K = n_size;
        if K ~= round(K)
            error('n_size must be integer for ''k''.');
        end
    elseif strcmpi(n_fcn, 'epsilon')
        epsilon = n_size;
    else
        error('n_fcn must be ''k'' or ''epsilon''.');
    end
    % Default options
    if nargin < 4 || isempty(optsIn), optsIn = struct(); end
    optsIn = setfieldIfMissing(optsIn, 'dims',        1:10);
    optsIn = setfieldIfMissing(optsIn, 'comp',        1);
    optsIn = setfieldIfMissing(optsIn, 'overlay',     0);
    optsIn = setfieldIfMissing(optsIn, 'verbose',     1);
    optsIn = setfieldIfMissing(optsIn, 'computeMode', 'gpu');
    dims        = optsIn.dims;
    comp        = optsIn.comp;
    overlay     = optsIn.overlay;
    verbose     = optsIn.verbose;
    computeMode = optsIn.computeMode;
    % Prepare outputs
    Y.coords = cell(length(dims),1);
    R        = zeros(1,length(dims));
    numPoints = size(D,1);
    INF = 1000 * max(D(:)) * numPoints;

%% 1: CONSTRUCT NEIGHBORHOOD GRAPH
    if verbose, fprintf('Constructing neighborhood graph...\n'); end
    switch lower(n_fcn)
        case 'k'
            [~, idx] = sort(D,1);
            for i = 1:numPoints
                D(i, idx((K+2):end,i)) = INF;
            end
        case 'epsilon'
            warning('off','all');
            D = D ./ (D <= epsilon);
            D = min(D, INF);
            warning('on','all');
    end
    D = min(D, D');  % ensure symmetry
    if overlay
        E = int8(1 - (D == INF));
    else
        E = [];
    end

%% 2: COMPUTE SHORTEST PATHS
    if strcmpi(computeMode,'gpu')
        if verbose, fprintf('Computing shortest paths on GPU...\n'); end
        Dg = gpuArray(D);
        for k = 1:numPoints
            Dg = min(Dg, Dg(:,k) + Dg(k,:));
        end
        D = gather(Dg);
    else
        if verbose, fprintf('Computing shortest paths on CPU...\n'); end
        for k = 1:numPoints
            D = min(D, repmat(D(:,k),1,numPoints) + repmat(D(k,:),numPoints,1));
        end
    end

%% 3: EXTRACT CONNECTED COMPONENT
    if verbose, fprintf('Extracting connected component...\n'); end
    [~, firstIdx] = min(D==INF,[],1);
    [comps, ~, compIdx] = unique(firstIdx);
    sizes = accumarray(compIdx(:),1);
    [~, order] = sort(sizes,'descend');
    comps = comps(order);
    if comp > numel(comps), comp = 1; end
    sel = comps(comp);
    Y.index = find(firstIdx == sel);
    D = D(Y.index, Y.index);
    numPoints = numel(Y.index);

%% 4: CLASSICAL MDS EMBEDDING
    if verbose, fprintf('Performing Classical MDS...\n'); end
    D2 = D.^2;
    m1 = mean(D2,2); m2 = mean(D2,1); M = mean(D2(:));
    H = -0.5 * (D2 - m1 - m2 + M);
    opts.disp = 0;
    [V, L] = eigs(H, max(dims), 'LR', opts);
    [lam, order] = sort(real(diag(L)),'descend');
    V = V(:,order);
    D_vec = D(:);

%% 5: COMPUTE COORDS & RESIDUAL VARIANCE
    for i = 1:length(dims)
        d = dims(i);
        if d <= numPoints
            coords = (V(:,1:d) .* sqrt(lam(1:d)'))';
            Y.coords{i} = coords;
            embD = pdist2(coords', coords');
            C = corrcoef(embD(:), D_vec);
            R(i) = 1 - C(2,1)^2;
            if verbose
                fprintf('Embedding %d dims: residual variance = %.6f\n', d, R(i));
            end
        end
    end

end

%% Helper: set default field if missing
function s = setfieldIfMissing(s, name, val)
    if ~isfield(s,name)
        s.(name) = val;
    end
end


%    BEGIN COPYRIGHT NOTICE
%
%    Isomap code -- (c) 1998-2000 Josh Tenenbaum
%
%    This code is provided as is, with no guarantees except that 
%    bugs are almost surely present.  Published reports of research 
%    using this code (or a modified version) should cite the 
%    article that describes the algorithm: 
%
%      J. B. Tenenbaum, V. de Silva, J. C. Langford (2000).  A global
%      geometric framework for nonlinear dimensionality reduction.  
%      Science 290 (5500): 2319-2323, 22 December 2000.  
%
%    Comments and bug reports are welcome.  Email to jbt@psych.stanford.edu. 
%    I would also appreciate hearing about how you used this code, 
%    improvements that you have made to it, or translations into other
%    languages.    
%
%    You are free to modify, extend or distribute this code, as long 
%    as this copyright notice is included whole and unchanged.  
%
%    END COPYRIGHT NOTICE
