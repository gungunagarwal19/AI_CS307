% eight_rook_full.m
clear; close all; clc;

N = 8; 
A = 4; 
B = 4;

n = N * N; 
W = zeros(n); 
b = zeros(n,1);

% Build weight matrix and bias
for i = 1:N
    for j = 1:N
        idx1 = (i-1)*N + j;

        for k = 1:N
            for l = 1:N
                idx2 = (k-1)*N + l;

                if idx1 == idx2
                    continue;
                end

                % Same row → penalty
                if i == k
                    W(idx1, idx2) = W(idx1, idx2) - 2*A;
                end

                % Same column → penalty
                if j == l
                    W(idx1, idx2) = W(idx1, idx2) - 2*B;
                end
            end
        end

        % Bias term
        b(idx1) = b(idx1) + 2*A + 2*B;
    end
end

% Initial state: one rook per row
state = zeros(n,1);
for i = 1:N
    j = randi(N);
    state((i-1)*N + j) = 1;
end

% Hopfield dynamics
for it = 1:2000
    idx = randi(n);
    h = W(idx,:) * state - b(idx);
    new = (h < 0);
    state(idx) = new;

    S = reshape(state, N, N);

    % Check valid solution: exactly 1 rook per row and col
    if all(sum(S,2)==1) && all(sum(S,1)==1)
        break;
    end
end

% Display
S = reshape(state, N, N);
imagesc(S);
colormap(gray(2));
title('Eight-Rook');
