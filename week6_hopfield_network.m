clear; close all; clc;

Nrows = 10;
Ncols = 10;
N = Nrows * Ncols;
P = 12;

rng(1);

X = randi([0 1], N, P);
X(X==0) = -1;

W = zeros(N, N);
for p = 1:P
    W = W + X(:, p) * X(:, p)';
end

W(1:N+1:end) = 0;
W = W / P;

x = X(:, 1);
idx = randperm(N, round(0.20 * N));
xnoisy = x;
xnoisy(idx) = -xnoisy(idx);

state = xnoisy;
maxit = 20000;

for it = 1:maxit
    i = randi(N);
    h = W(i, :) * state;
    state(i) = sign(h);
    if state(i) == 0, state(i) = 1; end
    if isequal(state, x), break; end
end

orig = reshape(X(:, 1), Nrows, Ncols)';
noisy = reshape(xnoisy, Nrows, Ncols)';
recalled = reshape(state, Nrows, Ncols)';

figure;
subplot(1, 3, 1); imshow(orig > 0); title('Original');
subplot(1, 3, 2); imshow(noisy > 0); title('Noisy (20%)');
subplot(1, 3, 3); imshow(recalled > 0); title('Recalled');
