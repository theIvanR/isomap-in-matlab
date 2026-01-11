function [Y, R] = Isomap(X, opts)
% ISOMAP  Chunked Isomap (CPU-only) for large datasets
%
%   [Y, R] = Isomap(X, opts)
%
% INPUTS:
%   X    : N x D data matrix (observations in rows)
%   opts : structure with fields (all optional)
%       .n_fcn  : 'k' (default) or 'epsilon' (neighborhood function)
%       .n_size : neighborhood parameter (k integer or epsilon scalar)
%       .dims   : vector of requested embedding dims (default 1:10)
%       .comp   : which connected component to embed by size rank (default 1 = largest)
%       .verbose: 1/0 print progress (default 1)
%       .chunk  : chunk size (rows per block) for pdist2 (default 1000)
%
% OUTPUTS:
%   Y.coords : cell array with coordinates (each cell is d x N_comp for each dims entry)
%   R        : residual variance for each requested embedding (1 x numel(dims))

if nargin < 1, error('Requires X input'); end
if nargin < 2, opts = struct(); end

% ---- defaults ----
opts = setIfMissing(opts, 'n_fcn', 'k');
opts = setIfMissing(opts, 'n_size', 10);
opts = setIfMissing(opts, 'dims', 1:10);
opts = setIfMissing(opts, 'comp', 1);
opts = setIfMissing(opts, 'verbose', 1);
opts = setIfMissing(opts, 'chunk', 1000);

n_fcn = opts.n_fcn;
n_size = opts.n_size;
dims = opts.dims;
comp = opts.comp;
verbose = opts.verbose;
chunkSize = max(1, round(opts.chunk));

[N, D] = size(X);
Y.coords = cell(length(dims),1);
R = nan(1,length(dims));

% ---- 1) build neighborhood sparse graph (chunked distance computation) ----
if verbose, fprintf('Building neighborhood graph (chunked)...\n'); end

switch lower(n_fcn)
    case 'k'
        K = n_size;
        if K ~= round(K), error('n_size must be integer for ''k'''); end
        kq = min(K+1, N);  % include self
        % preallocate arrays for K*N nonzero entries (upper bound)
        estNNZ = N * K;
        i_list = zeros(estNNZ,1,'uint32');
        j_list = zeros(estNNZ,1,'uint32');
        w_list = zeros(estNNZ,1,'double');
        ptr = 1;
        for i0 = 1:chunkSize:N
            i1 = min(i0 + chunkSize - 1, N);
            rows = i0:i1;
            if verbose, fprintf('  chunk %d:%d ...\n', i0, i1); end
            % compute distances for this block (rows x N)
            Dblock = pdist2(single(X(rows,:)), single(X), 'euclidean'); % chunk x N, single for memory
            % get k+1 neighbors then remove self
            [valsBlock, idxBlock] = mink(Dblock, kq, 2);
            % build mask to remove self-links
            srcRows = repmat(rows', 1, kq);
            keepMask = (idxBlock ~= srcRows);  % chunk x kq logical
            % flatten and keep only K entries per row (after removing self)
            % After removing self there will be exactly K entries per row when kq==K+1
            valsRow = valsBlock(:,1:kq);
            idxRow = idxBlock(:,1:kq);
            % pick only columns where keepMask true
            valsSel = valsRow(keepMask);
            idxSel = idxRow(keepMask);
            % source indices repeated
            numNew = numel(idxSel);
            if ptr + numNew - 1 > numel(i_list)
                % grow arrays (rare): double capacity
                newCap = max(numel(i_list)*2, ptr+numNew);
                i_list(end+1:newCap) = uint32(0);
                j_list(end+1:newCap) = uint32(0);
                w_list(end+1:newCap) = 0;
            end
            % fill
            % create src column matching idxSel
            repSrc = repmat(rows', 1, K)'; % not used directly, simpler to generate:
            % compute row indices corresponding to keepMask true
            [rInd, cInd] = find(keepMask); % rInd: row within chunk, cInd: neighbor column
            srcIndices = rows(rInd)';
            i_list(ptr:ptr+numNew-1) = uint32(srcIndices(:));
            j_list(ptr:ptr+numNew-1) = uint32(idxSel(:));
            w_list(ptr:ptr+numNew-1) = double(valsSel(:));
            ptr = ptr + numNew;
            clear Dblock valsBlock idxBlock valsRow idxRow valsSel idxSel keepMask rInd cInd srcIndices
        end
        % trim arrays
        i_list = double(i_list(1:ptr-1));
        j_list = double(j_list(1:ptr-1));
        w_list = double(w_list(1:ptr-1));
    case 'epsilon'
        epsv = n_size;
        % for epsilon it's hard to pre-estimate nnz; collect in cell arrays per chunk
        i_cells = {};
        j_cells = {};
        w_cells = {};
        total = 0;
        for i0 = 1:chunkSize:N
            i1 = min(i0 + chunkSize - 1, N);
            rows = i0:i1;
            if verbose, fprintf('  chunk %d:%d ...\n', i0, i1); end
            Dblock = pdist2(single(X(rows,:)), single(X), 'euclidean'); % chunk x N
            mask = (Dblock <= epsv) & (Dblock > 0);
            [rInd, cInd] = find(mask);
            if ~isempty(rInd)
                srcIndices = rows(rInd)';
                i_cells{end+1} = double(srcIndices(:));
                j_cells{end+1} = double(cInd(:));
                w_cells{end+1} = double(Dblock(mask));
                total = total + numel(rInd);
            end
            clear Dblock mask rInd cInd srcIndices
        end
        if total == 0
            i_list = [];
            j_list = [];
            w_list = [];
        else
            i_list = vertcat(i_cells{:});
            j_list = vertcat(j_cells{:});
            w_list = vertcat(w_cells{:});
        end
    otherwise
        error('n_fcn must be ''k'' or ''epsilon''');
end

% If no neighbors found (degenerate), fail
if isempty(i_list)
    error('No neighbors found — check n_size or data.');
end

% build sparse adjacency and symmetrize
A = sparse(i_list, j_list, double(w_list), N, N);
A = min(A, A'); % enforce symmetry and choose smaller weight
[iw, jw, vw] = find(A);
keep = (vw > 0) & isfinite(vw);
A = sparse(iw(keep), jw(keep), vw(keep), N, N);

% ---- 2) shortest-path geodesic distances ----
if verbose, fprintf('Computing all-pairs shortest paths using graph distances...\n'); end
G = graph(A);
Dgeo_full = distances(G);

% large sentinel for unreachable pairs (use max observed finite distance)
finiteWeights = vw(isfinite(vw) & vw>0);
if isempty(finiteWeights)
    maxW = 1;
else
    maxW = max(finiteWeights);
end
INF = 1e9 * maxW * max(1,N);

% ---- 3) connected components ----
if verbose, fprintf('Extracting connected components...\n'); end
compLabels = conncomp(G);
tab = histcounts(compLabels, 1:max(compLabels)+1);
[~, orderIdx] = sort(tab, 'descend');
orderIdx = orderIdx(:);
if comp > numel(orderIdx), comp = 1; end
selLabel = orderIdx(comp);
sel = find(compLabels == selLabel);
Y.index = sel; % indices in original data

% restrict geodesic distance to component
Dgeo = Dgeo_full(sel, sel);
Ncomp = numel(sel);

% ---- 4) classical MDS ----
if verbose, fprintf('Performing classical MDS on %d points...\n', Ncomp); end
% handle unreachable pairs
unreachable = ~isfinite(Dgeo) | (Dgeo > 0.5*INF);
if any(unreachable(:))
    Dgeo(unreachable) = INF;
end
D2 = (Dgeo).^2;
% double-centering
m1 = mean(D2,2); m2 = mean(D2,1); M = mean(D2(:));
H = -0.5 * (D2 - m1 - m2 + M);

kmax = max(dims);
opts_eigs.tol = 1e-6; opts_eigs.maxit = 500;
if Ncomp <= 2000
    [Vfull, Lfull] = eig(H);
    [lamAll, ord] = sort(real(diag(Lfull)),'descend');
    Vfull = Vfull(:, ord);
    V = Vfull(:, 1:min(kmax, Ncomp));
    lam = lamAll(1:min(kmax, Ncomp));
else
    try
        [V, L] = eigs(H, min(kmax, Ncomp-1), 'la', opts_eigs);
        [lam, order] = sort(real(diag(L)),'descend');
        V = V(:, order);
    catch
        warning('eigs failed, falling back to full eig');
        [Vfull, Lfull] = eig(H);
        [lamAll, ord] = sort(real(diag(Lfull)),'descend');
        Vfull = Vfull(:, ord);
        V = Vfull(:, 1:min(kmax, Ncomp));
        lam = lamAll(1:min(kmax, Ncomp));
    end
end

% clip tiny negatives due to numerical noise
lam(lam < 0) = 0;

% prepare original geodesic vector for residual correlation
Dgeo_vec = Dgeo(:);

for ii = 1:length(dims)
    d = dims(ii);
    if d <= Ncomp
        % coordinates: d x Ncomp (rows: axes)
        Lam_d = sqrt(lam(1:d));
        coords = (V(:,1:d) .* Lam_d')';
        Y.coords{ii} = coords;
        % compute pairwise distances from coords
        Xc = coords'; % Ncomp x d
        sq = sum(Xc.^2,2);
        Gdist2 = bsxfun(@plus, sq, sq') - 2*(Xc*Xc');
        Gdist2(Gdist2 < 0) = 0;
        embD = sqrt(Gdist2);
        % correlation-based residual variance
        C = corrcoef(embD(:), Dgeo_vec);
        if numel(C) < 4 || any(isnan(C(:)))
            R(ii) = NaN;
        else
            R(ii) = 1 - C(2,1)^2;
        end
        if verbose, fprintf('Embedding %d dims: residual variance = %.6f\n', d, R(ii)); end
    else
        Y.coords{ii} = [];
        R(ii) = NaN;
    end
end

if verbose, fprintf('Isomap finished.\n'); end
end

%% helper
function s = setIfMissing(s, name, val)
    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = val;
    end
end
