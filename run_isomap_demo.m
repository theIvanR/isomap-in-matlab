
%% 0: INTRODUCTION
% Generate a Swiss Roll dataset, compute pairwise distances,
% run Isomap for 2D embedding, and visualize both original and embedded data.

%% 1: GENERATE DATASET
N = 5000;                                   % Number of data points
t = (3*pi/2) * (1 + 2*rand(1, N));          % Spiral angle
height = 21 * rand(1, N);                   % Height component
X = [t.*cos(t); height; t.*sin(t)]';        % 3D Swiss Roll coordinates

%% 2: SET ISOMAP PARAMETERS
n_fcn = 'k';                                % Neighborhood function
n_size = 10;                                % Number of neighbors
options = struct( ...
    'dims', 2, ...
    'comp', 1, ...
    'verbose', 1, ...
    'chunk', 1000 ...
);

%% 3: RUN ISOMAP
fprintf('Starting Isomap on %d points...\n', N);
tic;
[Y, R] = isomap(X, options);                % Isomap embedding
elapsed = toc;
fprintf('Isomap finished in %.2f seconds.\n', elapsed);

%% 4: EXTRACT EMBEDDING
embedding = Y.coords{1}';                   % 2D coordinates

%% 5: PLOT RESULTS
figure('Position', [100 100 1000 500]);

subplot(1, 2, 1);
scatter3(X(:,1), X(:,2), X(:,3), 10, t, 'filled');
title('Original Swiss Roll');
colormap('jet');
colorbar;
view([-45 20]);

subplot(1, 2, 2);
scatter(embedding(:,1), embedding(:,2), 10, t(Y.index), 'filled');
title(sprintf('Isomap 2D Embedding (R^2 = %.4f)', 1-R(1)));
colormap('jet');
colorbar;
axis equal;

saveas(gcf, 'isomap_result.png');
