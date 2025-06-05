%% 0: INTRODUCTION
% Generate a Swiss Roll dataset, compute pairwise distances,
% run Isomap for 2D embedding, and visualize both original and embedded data.

%% 1: GENERATE DATASET
N = 5000;                                   % Number of data points
t = (3*pi/2) * (1 + 2*rand(1, N));          % Spiral angle
height = 21 * rand(1, N);                   % Height component
X = [t.*cos(t); height; t.*sin(t)]';        % 3D Swiss Roll coordinates

%% 2: COMPUTE DISTANCES
D = pdist2(X, X, 'fasteuclidean');          % Pairwise Euclidean distances

%% 3: SET ISOMAP PARAMETERS
n_fcn = 'k';                                % Neighborhood function
n_size = 10;                                % Number of neighbors
options = struct( ...
    'dims', 2, ...
    'comp', 1, ...
    'verbose', 1, ...
    'computeMode', 'cpu' ...
);

%% 4: RUN ISOMAP
[Y, R] = Isomap(D, n_fcn, n_size, options); % Isomap embedding

%% 5: EXTRACT EMBEDDING
embedding = Y.coords{1}';                   % 2D coordinates

%% 6: PLOT RESULTS
figure;

subplot(1, 2, 1);
scatter3(X(:,1), X(:,2), X(:,3), 10, t, 'filled');
title('Original Swiss Roll');
colormap('jet');
colorbar;
view([-45 20]);

subplot(1, 2, 2);
scatter(embedding(:,1), embedding(:,2), 10, t(Y.index), 'filled');
title('Isomap 2D Embedding');
colormap('jet');
colorbar;
axis equal;
