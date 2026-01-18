clear; clc;
M = 16;
K = 4*M;
theta = linspace(-90, 90, K);
phi = sind(theta);
A = zeros(K, M);
for k=1:K
    A(k,:) = exp(1j * pi * (0:M-1) * phi(k));
end

desired_dirs = [0];
v_mag = zeros(K,1);
dir_indices = round((desired_dirs + 90) / (180/(K-1)) + 1);
dir_indices = max(min(dir_indices, K), 1);
v_mag(dir_indices) = 1;
D = eye(K);
D_v = diag(v_mag); % as provided

% GA Parameters
pop_size = 100;
gens = 200;
cross_prob = 0.9;
mut_prob = 0.1;
elite_perc = 0.1; % 10% elites
num_elites = max(1, round(elite_perc * pop_size)); % ensure at least 1
chrom_len = 2*M;

% initialize population (each row is a chromosome)
pop = -1 + 2*rand(pop_size, chrom_len);

for gen=1:gens
    % Evaluate fitness for current population
    fitness = zeros(pop_size,1);
    for i=1:pop_size
        fitness(i) = eval_fitness(pop(i,:), A, D, D_v, K, M);
    end

    % Sort and grab elites (best first)
    [~, sort_idx] = sort(fitness, 'descend');
    elites = pop(sort_idx(1:num_elites), :);

    % Prepare offspring count
    num_offspring = pop_size - num_elites;
    % Ensure even number of offspring pairs for pairwise reproduction
    pairs = floor(num_offspring / 2);

    new_pop = zeros(num_offspring, chrom_len);
    pos = 1;
    for p = 1:pairs
        % Parent 1 selection via tournament on global indices
        cand1 = randi(pop_size,2,1);
        win1_pos = tournament(fitness(cand1)); % 1 or 2
        win1_global = cand1(win1_pos);
        p1 = pop(win1_global, :);

        % Parent 2 selection
        cand2 = randi(pop_size,2,1);
        win2_pos = tournament(fitness(cand2));
        win2_global = cand2(win2_pos);
        p2 = pop(win2_global, :);

        % Crossover
        if rand < cross_prob
            [c1, c2] = simple_crossover(p1, p2);
        else
            c1 = p1; c2 = p2;
        end

        % Mutation
        if rand < mut_prob
            c1 = gauss_mutation(c1);
        end
        if rand < mut_prob
            c2 = gauss_mutation(c2);
        end

        new_pop(pos, :) = c1;
        new_pop(pos+1, :) = c2;
        pos = pos + 2;
    end

    % If we have an odd leftover slot (num_offspring odd), fill it with a mutated copy
    if num_offspring > 2*pairs
        % choose one parent by tournament and insert mutated copy
        cand = randi(pop_size,2,1);
        win_pos = tournament(fitness(cand));
        win_global = cand(win_pos);
        extra = pop(win_global, :);
        if rand < mut_prob
            extra = gauss_mutation(extra);
        end
        new_pop(end, :) = extra;
    end

    % Form next generation: elites first (preserve), then offspring
    pop = [elites; new_pop];
end

% Re-evaluate final fitness and pick best
fitness = zeros(pop_size,1);
for i=1:pop_size
    fitness(i) = eval_fitness(pop(i,:), A, D, D_v, K, M);
end
[~, best_idx] = max(fitness);
best_chrom = pop(best_idx,:);
best_w_real = best_chrom(1:M);
best_w_imag = best_chrom(M+1:end);
best_w = (best_w_real + 1j*best_w_imag).'; % make column vector (Mx1)
if norm(best_w) ~= 0
    best_w = best_w / norm(best_w);
end

pattern = abs(A * best_w);
% avoid log of zero; replace zeros by a very small number
pattern_db = 20*log10(max(pattern, 1e-12)/max(pattern));
figure; plot(theta, pattern_db); xlabel('Equivalent Direction'); ylabel('Gain (dB)');
title('Optimized Beam Pattern with Elitism');
disp('Best w:'); disp(best_w);

% Local functions

function fitness = eval_fitness(chrom, A, D, D_v, K, M)
    % convert chrom (row) to complex column vector
    w_real = chrom(1:M);
    w_imag = chrom(M+1:2*M);
    w = (w_real + 1j * w_imag).';   % column vector Mx1
    nrm = norm(w);
    if nrm == 0
        fitness = -norm(D_v * ones(K,1))^2;
        return;
    end
    w = w / nrm;

    % compute D * (A * w)
    Aw = A * w;              % Kx1
    DAw = D * Aw;            % Kx1

    denom = (norm(DAw)^2);
    if denom == 0
        cs = 0;
    else
        cs = (DAw' * (D * (D_v * ones(K,1)))) / denom; % scalar
    end

    % error: norm(D * (cs * A * w - D_v * ones(K,1)))^2
    target = D_v * ones(K,1);
    residual = D * (cs * (A * w) - target);
    error = norm(residual)^2;
    fitness = -error;
end

function [child1, child2] = simple_crossover(p1, p2)
    alpha = rand;
    child1 = alpha*p1 + (1-alpha)*p2;
    child2 = (1-alpha)*p1 + alpha*p2;
    child1 = max(min(child1,1),-1);
    child2 = max(min(child2,1),-1);
end

function child = gauss_mutation(child)
    child = child + 0.1 * randn(size(child));
    child = max(min(child,1),-1);
end

function idx = tournament(fit_values)
    % fit_values is a small vector (e.g., 2x1) of candidate fitness values
    % returns the index (1..length(fit_values)) of the winner
    [~, idx] = max(fit_values);
    idx = idx(1);
end