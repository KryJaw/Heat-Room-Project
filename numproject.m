clear;
clc;
close all;

% grid setup
nx = 20;
ny = 20;
N = nx * ny;
alpha = 0.5; 
dt = 0.1; 
dx = 1;
dy = 1; 
steps = 200; 
lambda = (alpha * dt) / dx^2; % handles how much neighboring points influence each other

% setting up the implicit matrix math
% using formula: (1 + 4*lambda)*T_i,j - lambda*(neighbors) = T_old
main_diag = ones(N, 1) * (1 + 4*lambda);
off_diag = ones(N, 1) * (-lambda);

% blending center and neighbor diagonals into matrix A
A = spdiags([off_diag, off_diag, main_diag, off_diag, off_diag], ...
 [-nx, -1, 0, 1, nx], N, N);

% pinning down the boundaries so room walls stay constant
is_boundary = false(nx, ny);
is_boundary(1,:) = true; % left wall
is_boundary(end,:) = true; % right wall
is_boundary(:,1) = true; % bottom wall
is_boundary(:,end) = true; % top wall
boundary_indices = find(is_boundary);

% replacing boundary equations so T_new just equals T_old on the edges
for idx = boundary_indices'
 A(idx, :) = 0;
 A(idx, idx) = 1;
end

% setting up starting conditions
T_initial = zeros(nx, ny);
T_initial(10, 10) = 100; % spiking a hot spot right in the middle
tol = 1e-6; % tolerance limit for when the iterative math gets close enough
max_iter = 10000; % safety cap to keep loop from running forever


%% GAUSSIAN ELIMINATION
T_vec_gauss = T_initial(:); % flattening to a vector for the solver
fprintf('Running Gaussian Solver...\n');
tic;
for k = 1:steps
 T_vec_gauss = A \ T_vec_gauss;
end
time_gauss = toc;

fprintf('\n--- RESULTS ---\n');
fprintf('Gaussian Time: %.4f seconds\n', time_gauss);

T_final_gauss = reshape(T_vec_gauss, nx, ny);
figure;
imagesc(T_final_gauss);
colorbar;
colormap(hot);
title(['Heat Map (Step ', num2str(steps), ')']);
xlabel('X-axis');
ylabel('Y-axis');


%% LU FACTORIZATION
T_vec_lu = T_initial(:); % flattening for solver
fprintf('Running LU Solver...\n');

% stripping A down to L and U once since it does not change across time steps
[L, U, P] = lu(A);
tic;
for k = 1:steps
 b = T_vec_lu;
% forward substitution step
 y = L \ (P * b);
% backward substitution step
 T_vec_lu = U \ y;
end
time_lu = toc;
fprintf('LU Time: %.4f seconds\n', time_lu);

T_final_lu = reshape(T_vec_lu, nx, ny);
figure;
imagesc(T_final_lu);
colorbar;
colormap(hot);
title(['Heat Map using LU Factorization (Step ', num2str(steps), ')']);
xlabel('X-axis');
ylabel('Y-axis');


%% JACOBI METHOD (FIXED)
T_vec_jacobi = T_initial(:); % flattening for solver
fprintf('Running Jacobi Solver...\n');

% grabbing the main diagonal and leaving the leftovers in R
diag_A = diag(A);
R = A - spdiags(diag_A, 0, N, N);
jacobi_iterations_total = 0;
tic;
for k = 1:steps
 b = T_vec_jacobi;
 x_old = T_vec_jacobi;
 
 for iter = 1:max_iter
     % vector math trick using element-wise division to calculate everything at once
     x_new = (b - R * x_old) ./ diag_A;
     
     % stopping early if things stop changing significantly
     if max(abs(x_new - x_old)) < tol
        break;
     end
     x_old = x_new;
 end
 jacobi_iterations_total = jacobi_iterations_total + iter;
 
 % fixed: remembering to pass the new calculations forward to the next time step
 T_vec_jacobi = x_new; 
end
time_jacobi = toc;
fprintf('Jacobi Time: %.4f seconds\n', time_jacobi);
fprintf('Total Jacobi Iterations: %d\n', jacobi_iterations_total);

T_final_jacobi = reshape(T_vec_jacobi, nx, ny);
figure;
imagesc(T_final_jacobi);
colorbar;
colormap(hot);
title(['Heat Map using Jacobi Method (Step ', num2str(steps), ')']);
xlabel('X-axis');
ylabel('Y-axis');


%% GAUSSE-SEIDEL METHOD (RESTORED TO ORIGINAL SCALAR LOOPS)
T_vec_seidel = T_initial(:); % flattening matrix to vector
fprintf('Running Gauss-Seidel Solver...\n');
seidel_iterations_total = 0; 
tic;
for k = 1:steps
 b = T_vec_seidel; % keeping track of current temperatures
 x_old = T_vec_seidel; % tracking last iteration state
 x_new = x_old; % priming new container
 
 for iter = 1:max_iter
    % un-vectorized row loop so it takes a couple seconds like in our report discussion
    for i = 1:N
        % grabbing parts we already calculated this round
        sum1 = A(i, 1:i-1) * x_new(1:i-1);
        % grabbing parts we have not gotten to yet
        sum2 = A(i, i+1:N) * x_old(i+1:N);
        % standard item-by-item formula update
        x_new(i) = (b(i) - sum1 - sum2) / A(i,i);
    end
    
    % escaping out if solver hits steady state
    if max(abs(x_new - x_old)) < tol
        break;
    end
    % tracking updates for next iteration loop
    x_old = x_new;
 end
 % stacking up total iterations across all time intervals
 seidel_iterations_total = seidel_iterations_total + iter;
 % sending updated values forward to next timestamp
 T_vec_seidel = x_new;
end
time_seidel = toc;
fprintf('Gauss-Seidel Time: %.4f seconds\n', time_seidel);
fprintf('Total Gauss-Seidel Iterations: %d\n', seidel_iterations_total);

% turning flat vector back into 2D grid for plot display
T_final_seidel = reshape(T_vec_seidel, nx, ny);
figure;
imagesc(T_final_seidel);
colorbar;
colormap(hot);
title(['Heat Map using Gauss-Seidel Method (Step ', num2str(steps), ')']);
xlabel('X-axis');
ylabel('Y-axis');


%% --- RESTORED ORIGINAL COMPARISON FORMATTING (%.10f) ---
diff_gauss_lu = max(abs(T_vec_gauss - T_vec_lu));
diff_gauss_jacobi = max(abs(T_vec_gauss - T_vec_jacobi));
diff_gauss_seidel = max(abs(T_vec_gauss - T_vec_seidel));

fprintf('\n--- COMPARISON ---\n');
fprintf('Difference Gaussian vs LU: %.10f\n', diff_gauss_lu);
fprintf('Difference Gaussian vs Jacobi: %.10f\n', diff_gauss_jacobi);
fprintf('Difference Gaussian vs Gauss-Seidel: %.10f\n', diff_gauss_seidel);

% plotting 3D peak visual
figure;
surf(T_final_gauss);
shading interp;
colormap(hot);
colorbar;
title('3D Surface Plot of Heat Distribution');
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Temperature');

% building error comparison maps
diff_grid_jacobi = abs(T_final_gauss - T_final_jacobi);
diff_grid_seidel = abs(T_final_gauss - T_final_seidel);

figure('Name', 'Monochrome Solver Error Maps', 'Position', [150, 150, 1200,500]);

% splitting layout to check jacobi error profile in grayscale
subplot(1, 2, 1);
imagesc(diff_grid_jacobi);
axis square;
colorbar;
colormap(gca, gray); 
title('Error Magnitude: |Gaussian - Jacobi|');
xlabel('X-axis'); ylabel('Y-axis');

% checking gauss-seidel directional bias in grayscale
subplot(1, 2, 2);
imagesc(diff_grid_seidel);
axis square;
colorbar;
colormap(gca, gray); 
title('Error Magnitude: |Gaussian - Gauss-Seidel|');
xlabel('X-axis'); ylabel('Y-axis');
