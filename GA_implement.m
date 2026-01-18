clear; clc;
M = 16; % Number of antennas
K = 4*M; % Directions
theta = linspace(-90, 90, K); % Degrees
phi = sind(theta); % sin(theta) for array response
A = zeros(K, M);
for k=1:K
    A(k,:) = exp(1j * pi * (0:M-1) * phi(k));
end

% Desired v magnitude: 1 at desired directions, 0 elsewhere
desired_dirs = [0]; % Communication and scanning example
v_mag = zeros(K,1);
dir_indices = round((desired_dirs + 90) / (180/(K-1)) + 1); % Map to indices
dir_indices = max(min(dir_indices, K), 1); % clamp just in case
v_mag(dir_indices) = 1;

% GA Parameters
pop_size = 100;
gens = 200;
cross_prob = 0.9;
mut_prob = 0.1;
eta_c = 20; % SBX eta
eta_m = 20; % Mutation eta
chrom_len = 2*M;

% Initialize population (each row is one chromosome, row vector)
pop = -1 + 2*rand(pop_size, chrom_len);

for gen=1:gens
    fitness = zeros(pop_size,1);
    for i=1:pop_size
        fitness(i) = eval_fitness(pop(i,:), A, v_mag, K, M);
    end

    % Tournament selection + reproduction
    new_pop = zeros(pop_size, chrom_len);
    for pair=1:(pop_size/2)
        % pick two candidates for parent1
        cand1 = randi(pop_size,2,1);
        winner1_global_idx = cand1(tournament(fitness(cand1))); % map to global index
        p1 = pop(winner1_global_idx, :);

        % pick two candidates for parent2
        cand2 = randi(pop_size,2,1);
        winner2_global_idx = cand2(tournament(fitness(cand2)));
        p2 = pop(winner2_global_idx, :);

        if rand < cross_prob
            [c1, c2] = sbx_crossover(p1, p2, eta_c);
        else
            c1 = p1; c2 = p2;
        end
        if rand < mut_prob
            c1 = poly_mutation(c1, eta_m);
            c2 = poly_mutation(c2, eta_m);
        end
        new_pop(2*pair-1:2*pair, :) = [c1; c2];
    end

    pop = new_pop;
end

% Evaluate final fitness to pick best
fitness = zeros(pop_size,1);
for i=1:pop_size
    fitness(i) = eval_fitness(pop(i,:), A, v_mag, K, M);
end
[~, best_idx] = max(fitness);
best_chrom = pop(best_idx,:);
best_w_real = best_chrom(1:M);
best_w_imag = best_chrom(M+1:end);
best_w = (best_w_real + 1j*best_w_imag).'; % column vector
best_w = best_w / norm(best_w);

% Plot beam pattern
pattern = abs(A * best_w);
figure; plot(theta, 20*log10(pattern/max(pattern))); xlabel('Equivalent Direction'); ylabel('Gain (dB)');
title('Optimized Beam Pattern with SBX GA');
disp('Best w:'); disp(best_w);

% Local functions (kept after main script)

function fitness = eval_fitness(chrom, A, v_mag, K, M)
    % convert chromosome to complex column vector w (M x 1)
    w_real = chrom(1:M);
    w_imag = chrom(M+1:2*M);
    w = (w_real + 1j * w_imag).';   % <-- make column vector (Mx1)
    if norm(w) == 0
        fitness = -norm(v_mag)^2;
        return;
    end
    w = w / norm(w); % Normalize power constraint
    y = A * w;       % A (KxM) * w (Mx1) -> y (Kx1)
    z = abs(y);
    d = v_mag;
    if all(z == 0)
        error_val = norm(d)^2;
    else
        denom = (z' * z);
        if denom == 0
            s = 0;
        else
            s = (d' * z) / denom; % optimal scalar scale
        end
        error_val = norm(s * z - d)^2;
    end
    fitness = -error_val;
end

function [child1, child2] = sbx_crossover(parent1, parent2, eta_c)
    % parents are row vectors
    r = rand(size(parent1));
    gamma = zeros(size(parent1));
    idx = r <= 0.5;
    gamma(idx) = (2*r(idx)).^(1/(eta_c+1));
    gamma(~idx) = (1 ./ (2*(1-r(~idx)))).^(1/(eta_c+1));
    child1 = 0.5 * ((1 + gamma).*parent1 + (1 - gamma).*parent2);
    child2 = 0.5 * ((1 - gamma).*parent1 + (1 + gamma).*parent2);
    % Bound to [-1,1]
    child1 = max(min(child1,1),-1);
    child2 = max(min(child2,1),-1);
end

function child = poly_mutation(child, eta_m)
    r = rand(size(child));
    delta = zeros(size(child));
    idx = r < 0.5;
    delta(idx) = (2*r(idx)).^(1/(eta_m+1)) - 1;
    delta(~idx) = 1 - (2*(1-r(~idx))).^(1/(eta_m+1));
    child = child + delta;
    child = max(min(child,1),-1);
end

function idx = tournament(fit_values)
    % fit_values is a small column vector of candidate fitnesses (e.g. 2x1)
    % returns the index (1..length(fit_values)) of the winner
    [~, idx] = max(fit_values);
    idx = idx(1); % ensure scalar
end