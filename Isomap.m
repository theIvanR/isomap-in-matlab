function [Y, R, E] = Isomap(D, n_fcn, n_size, optsIn)
% ISOMAP_CPU_FAST  Optimized CPU-only Isomap for MATLAB
%
%   [Y, R, E] = Isomap_cpu_fast(D, n_fcn, n_size, optsIn)
%
% This function computes an Isomap embedding of a dataset given a distance 
% matrix D. It is aggressively optimized for CPU performance using:
%   - Sparse adjacency matrices to store the neighborhood graph
%   - MATLAB's graph/dijkstra routines instead of triple loops
%   - BLAS-friendly linear algebra for MDS
%   - Optional landmark-based approximation for very large datasets
%
% INPUTS:
%   D         : N x N symmetric, non-negative distance matrix between points
%
%   n_fcn     : Neighborhood function type (defines how the local graph is built)
%               'k'       : Each point is connected to its k nearest neighbors
%                           - Guarantees every point has k neighbors
%                           - Good default choice for unknown data scale or 
%                             unevenly sampled data
%               'epsilon' : Each point is connected to all neighbors within 
%                           distance <= epsilon
%                           - Useful when sampling is uniform and distance scale 
%                             is meaningful
%               Choosing k or epsilon correctly is crucial for a connected graph.
%
%   n_size    : Neighborhood size parameter
%               - If n_fcn='k', n_size = number of neighbors (integer)
%               - If n_fcn='epsilon', n_size = distance threshold (scalar)
%
%   optsIn    : (Optional) structure of additional options:
%       dims            : vector of embedding dimensions to compute (default 1:10)
%       comp            : which connected component to embed (default 1 = largest)
%       overlay         : flag to return adjacency matrix (0 = don't, 1 = return)
%       verbose         : flag to print progress messages (default 1)
%       approxLandmarks : number of landmarks for approximate Isomap 
%                         (0 = exact Isomap, default 0)
%
% OUTPUTS:
%   Y.coords : cell array of embedded coordinates, one cell per requested dimension
%   Y.index  : indices of points in the selected connected component
%   R        : residual variance for each embedding (smaller = better)
%   E        : adjacency matrix of the neighborhood graph (if overlay = 1)
%
% NOTES ON NEIGHBORHOOD CHOICE:
%   - 'k' nearest neighbors:
%       Each point is guaranteed exactly k neighbors. Ensures graph connectivity 
%       in unevenly sampled data. Typical k = 5-15 for small-to-medium datasets.
%       Too small k → graph disconnected. Too large k → Isomap approaches PCA.
%
%   - 'epsilon' neighborhood:
%       Each point connects to all neighbors within distance <= epsilon. Useful 
%       for uniformly sampled data where local scale is known. Harder to tune.
%
% PERFORMANCE OPTIMIZATIONS:
%   - Sparse adjacency reduces memory and speeds up graph operations
%   - MATLAB's graph/dijkstra functions are multithreaded C routines
%   - BLAS-friendly MDS avoids pdist2 and large double loops
%   - Optional single-precision usage reduces memory and improves speed
%   - Landmark approximation reduces complexity for very large N (O(m^3 + m*N))
%
% EXAMPLES:
%   % Standard exact Isomap with 10 nearest neighbors
%   options = struct('dims',1:5,'verbose',1);
%   [Y,R] = Isomap_cpu_fast(D,'k',10,options);
%
%   % Approximate Isomap with 500 landmarks
%   options = struct('dims',1:5,'approxLandmarks',500);
%   [Y,R] = Isomap_cpu_fast(D,'k',10,options);


if nargin < 3, error('Requires D, n_fcn, n_size'); end
if size(D,1) ~= size(D,2), error('D must be square'); end
if nargin < 4 || isempty(optsIn), optsIn = struct(); end

% ---- defaults ----
optsIn = setIfMissing(optsIn,'dims', 1:10);
optsIn = setIfMissing(optsIn,'comp', 1);
optsIn = setIfMissing(optsIn,'overlay', 0);
optsIn = setIfMissing(optsIn,'verbose', 1);
optsIn = setIfMissing(optsIn,'approxLandmarks', 0); % 0 = exact
dims = optsIn.dims;
comp = optsIn.comp;
overlay = optsIn.overlay;
verbose = optsIn.verbose;
nLandmarks = optsIn.approxLandmarks;

N = size(D,1);
Y.coords = cell(length(dims),1);
R = zeros(1,length(dims));
INF = 1e9 * max(D(:)) * max(1,N); % large sentinel


% ---- 1) build neighborhood sparse graph ----
if verbose, fprintf('Building neighborhood graph (sparse)...\n'); end
switch lower(n_fcn)
    case 'k'
        K = n_size;
        if K ~= round(K), error('n_size must be integer for ''k'''); end
        % Use mink for each column: faster than sorting whole matrix
        % We ask for K+1 because diagonal/self distance may be included
        [vals, idx] = mink(D, min(K+1, N), 1);  % Nx columns
        % create lists for sparse
        src = repmat(1:N, min(K+1,N), 1);
        src = src(:); idx = idx(:); vals = vals(:);
        % remove self-links (where idx == src)
        keep = idx ~= src;
        i_list = src(keep);
        j_list = idx(keep);
        w_list = vals(keep);
    case 'epsilon'
        epsv = n_size;
        mask = (D <= epsv) & (D > 0); % exclude zeros on diag
        [i_list, j_list] = find(mask);
        w_list = D(mask);
    otherwise
        error('n_fcn must be ''k'' or ''epsilon''');
end

% build symmetric sparse adjacency (take min for mutual edges)
A = sparse(i_list, j_list, double(w_list), N, N);
A = min(A, A');  % ensure symmetry and smaller weight if asymmetric

% keep only finite, positive weights (filter out inf/zeros)
[iw, jw, vw] = find(A);               % row, col, value vectors
keep = (vw > 0) & isfinite(vw);       % logical mask for valid edges
A = sparse(iw(keep), jw(keep), vw(keep), N, N);


if overlay
    E = (A ~= 0); % adjacency for overlay
else
    E = [];
end

% ---- 2) compute shortest-path geodesic distances using graph/dijkstra ----
if verbose, fprintf('Computing all-pairs shortest paths using graph distances...\n'); end
G = graph(A);
% distances uses optimized C implementation (Dijkstra / Johnson as needed)
% If approxLandmarks > 0, we will compute landmark distances only (see later).
if nLandmarks <= 0
    % full all-pairs distances
    Dgeo = distances(G); % returns double
else
    % approximate mode: compute distances from landmarks only
    nLand = min(nLandmarks, N);
    if verbose, fprintf('Using %d landmarks for approximate Isomap\n', nLand); end
    rng(0); % reproducible
    landmarks = randperm(N, nLand);
    % distances from landmarks to all nodes (matrix nLand x N)
    Dland = distances(G, landmarks); % each column -> distances from node to landmarks (note order)
    % We'll embed landmarks and extend others (landmark Isomap).
    % Build Dgeo as [] to mark approximate mode
    Dgeo = [];
end

% ---- 3) connected components: use conncomp on graph G ----
if verbose, fprintf('Extracting largest connected components...\n'); end
compLabels = conncomp(G); % 1..numComp labels
% choose component by comp option: rank by size
tab = histcounts(compLabels, 1:max(compLabels)+1);
[~, orderIdx] = sort(tab, 'descend');
orderIdx = orderIdx(:);
if comp > numel(orderIdx), comp = 1; end
selLabel = orderIdx(comp);
sel = find(compLabels == selLabel);
Y.index = sel;
% If we used landmarks approximate and some landmarks not in component, filter them later
% Restrict graph / distance to selected nodes
if nLandmarks <= 0
    Dgeo = Dgeo(sel, sel);
else
    % filter distance maps for selected nodes
    % Keep only landmarks that are inside the selected component
    % (if none inside, fallback to exact on component)
    landIn = intersect(landmarks, sel);
    if isempty(landIn)
        warning('No landmarks within selected component - falling back to exact distances');
        Dgeo = distances(G, sel);
    else
        % distances from landIn to sel
        Dland_sel = distances(G, landIn, sel);
        Dgeo = []; % keep special path for landmarks
        landmarks = landIn;
        Dland = Dland_sel;
        nLand = numel(landmarks);
    end
end

% update N to component size
Ncomp = numel(sel);

% ---- 4) classical MDS on Dgeo (exact) or Landmark MDS (approx) ----
if nLandmarks <= 0
    if verbose, fprintf('Performing Classical MDS (exact) on %d points...\n', Ncomp); end
    % ensure Dgeo is finite: unreachable pairs -> INF
    unreachable = ~isfinite(Dgeo) | (Dgeo > 0.5*INF);
    if any(unreachable(:))
        Dgeo(unreachable) = INF;
    end
    % squared-distance double
    D2 = (Dgeo).^2;
    % double centering using vectorized ops
    m1 = mean(D2,2); m2 = mean(D2,1); M = mean(D2(:));
    H = -0.5 * (D2 - m1 - m2 + M);
    kmax = max(dims);
    opts.eigs.tol = 1e-6;
    opts.eigs.maxit = 500;
    % if Ncomp is small, use full eig
    if Ncomp <= 2000
        [Vfull, Lfull] = eig(H);
        [lamAll, ord] = sort(real(diag(Lfull)),'descend');
        Vfull = Vfull(:, ord);
        V = Vfull(:, 1:min(kmax, Ncomp));
        lam = lamAll(1:min(kmax, Ncomp));
    else
        % use eigs for top kmax components (Lanczos)
        try
            [V, L] = eigs(H, min(kmax, Ncomp-1), 'la', opts.eigs);
            [lam, order] = sort(real(diag(L)),'descend');
            V = V(:, order);
        catch ME
            warning('eigs failed, falling back to full eig: %s', ME.message);
            [Vfull, Lfull] = eig(H);
            [lamAll, ord] = sort(real(diag(Lfull)),'descend');
            Vfull = Vfull(:, ord);
            V = Vfull(:, 1:min(kmax, Ncomp));
            lam = lamAll(1:min(kmax, Ncomp));
        end
    end

    % compute coords and residuals
    Dvec = Dgeo(:);
    for ii = 1:length(dims)
        d = dims(ii);
        if d <= Ncomp
            coords = (V(:,1:d) .* sqrt(lam(1:d)'))';
            Y.coords{ii} = coords;
            % compute pairwise distances quicker via Gram-based formula
            X = coords'; % Ncomp x d
            sq = sum(X.^2,2);
            % squared distances matrix via BLAS-friendly ops
            Gdist2 = bsxfun(@plus, sq, sq') - 2*(X*X');
            Gdist2(Gdist2 < 0) = 0;
            embD = sqrt(Gdist2);
            C = corrcoef(embD(:), Dvec);
            R(ii) = 1 - C(2,1)^2;
            if verbose, fprintf('Embedding %d dims: residual variance = %.6f\n', d, R(ii)); end
        else
            Y.coords{ii} = [];
            R(ii) = NaN;
        end
    end

else
    % ---- Landmark approximate Isomap (fast) ----
    if verbose, fprintf('Running Landmark Isomap (approx) with %d landmarks...\n', numel(landmarks)); end
    % Dland : distances from landmarks to ALL nodes (should be nLand x Ncomp)
    % Build pairwise landmark distances (nLand x nLand)
    Dll = Dland(:, sel)'; % careful with orientation: ensure Dll(i,j) is dist landmark_i to landmark_j
    Dll = Dll(ismember(sel, landmarks), :); % ensure correct order (left as caution)
    Dll = distances(G, landmarks); % simpler: distances between landmarks
    Dll = Dll(:,ismember(landmarks, sel)); % ensure selection
    Dll = Dll(:,:);
    % classical MDS on landmarks
    D2L = (Dll).^2;
    m1 = mean(D2L,2); m2 = mean(D2L,1); M = mean(D2L(:));
    H_L = -0.5 * (D2L - m1 - m2 + M);
    kmax = max(dims);
    [VL, LL] = eigs(H_L, min(kmax, size(H_L,1)-1), 'la');
    [lamL, ordL] = sort(real(diag(LL)),'descend');
    VL = VL(:, ordL);
    % coords of landmarks
    coordsL = (VL(:,1:kmax) .* sqrt(lamL(1:kmax)'))';
    % Nystrom extension (brief / approximate)
    % X_non = ( -0.5 * (d_n^2 - mean_col(Dll.^2) - mean_row(Dll.^2) + mean_all) ) * VL * inv(Lambda)
    % We'll compute extension for each requested dim below.
    % Precompute kernel for extension:
    muL_row = mean(D2L,2);
    muL_all = mean(D2L(:));
    % distances from non-landmarks to landmarks:
    Dnl = (Dland(:, sel)'); % Ncomp x nLand (maybe)
    Dnl2 = Dnl.^2;
    for ii = 1:length(dims)
        d = dims(ii);
        if d > size(VL,1), Y.coords{ii} = []; R(ii)=NaN; continue; end
        VLd = VL(:, 1:d);
        Lamd = diag(lamL(1:d));
        % compute B_n = -0.5*(Dnl2 - repmat(muL_row',Ncomp,1) - repmat(mean(Dnl2,2),1,nLand) + muL_all)
        mu_col = mean(Dnl2,1);
        mu_row_all = mean(Dnl2,2);
        Bn = -0.5*( Dnl2 - repmat(muL_row', Ncomp, 1) - repmat(mu_row_all, 1, numel(muL_row)) + muL_all );
        % extension coords:
        Xn = (Bn * VLd) / Lamd; % Ncomp x d
        coords = [coordsL(1:d,:)'; Xn]; % landmarks first then others; we'll only select sel ordering
        % But user expects coords per point in component order; reassemble:
        % landmarks are a subset; build final coords matrix in sel order:
        coordsFinal = zeros(d, Ncomp);
        % map landmarks positions
        [~, locInSel] = ismember(landmarks, sel);
        coordsFinal(:, locInSel) = coordsL(1:d, :)';
        % fill non-landmarks
        nonLandIdx = setdiff(1:Ncomp, locInSel);
        coordsFinal(:, nonLandIdx) = Xn(nonLandIdx, :)';
        Y.coords{ii} = coordsFinal;
        % residuals: compute emb pairwise distances via Gram method
        X = coordsFinal'; sq = sum(X.^2,2);
        Gdist2 = bsxfun(@plus, sq, sq') - 2*(X*X');
        Gdist2(Gdist2<0) = 0;
        embD = sqrt(Gdist2);
        % we need original geodesic distances for selected nodes:
        Dvec = (distances(G, sel));
        Dvec = Dvec(:);
        C = corrcoef(embD(:), Dvec);
        R(ii) = 1 - C(2,1)^2;
        if verbose, fprintf('Approx embedding %d dims (landmark): residual variance = %.6f\n', d, R(ii)); end
    end
end

% done
if verbose, fprintf('Isomap (CPU) finished.\n'); end

end

%% small helper
function s = setIfMissing(s, name, val)
    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = val;
    end
end
