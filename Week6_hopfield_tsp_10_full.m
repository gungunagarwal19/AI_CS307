% hopfield_tsp_10_full.m
clear; close all; clc;

N = 10; 
rng(2);

% Generate random city coordinates and distances
coords = rand(N,2) * 100;
d = pdist2(coords, coords);

% Hopfield-Tank parameters
A = 500; 
B = 500; 
C = 1; 
D = 0.01;
max_iter = 3000;

% Initial continuous state matrix
V = rand(N, N) * 0.1;

% Hopfield dynamics
for iter = 1:max_iter
    U = zeros(N, N);

    for x = 1:N
        for i = 1:N

            % Constraint violations
            row_term = A * (sum(V(x, :)) - 1);
            col_term = B * (sum(V(:, i)) - 1);

            % Next position (circular)
            next = mod(i, N) + 1;

            % Distance-based energy
            dist_term = sum(d(x, :) .* V(:, next)');

            % Total input
            U(x, i) = -row_term - col_term - C * dist_term;
        end
    end

    % Update neurons using sigmoid
    V = 1 ./ (1 + exp(-U / D));

    % Optional row normalization (ensures close-to one-hot)
    V = V ./ sum(V, 2);
end

% Extract final tour (taking max activation per row)
[~, pos] = max(V, [], 2);

tour = zeros(N,1);
for x = 1:N
    tour(pos(x)) = x;
end

% Compute tour length
len = 0;
for i = 1:N
    a = tour(i);
    bcity = tour(mod(i, N) + 1);
    len = len + d(a, bcity);
end

fprintf('Tour length (approx): %.3f\n', len);

% Plot final activation matrix
figure;
imagesc(V);
colorbar;
title('Final V (continuous)');
