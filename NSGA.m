function nsga2_multibeam_complete()
    clc; rng(1);

    %% ====================== PARAMETERS ===========================
    M = 16;
    lambda = 1;
    d = lambda/2;
    k = 2*pi/lambda;

    theta_com = 0;
    theta_scan = -60:1:60;
    theta_scan(theta_scan == theta_com) = [];

    theta_full = -90:0.2:90;   % for plotting beam patterns

    popSize = 80;
    maxGen  = 120;

    pC = 0.9;
    etaC = 20;
    pM = 1/(2*M);
    etaM = 20;

    dim = 2*M;

    %% ===================== INITIAL POP ===========================
    pop = 2*rand(popSize, dim) - 1;
    pop = normalize_population(pop, M);

    %% ================== EVOLUTION ===========================
    fprintf('Starting NSGA-II...\n');

    for gen = 1:maxGen

        % evaluate
        F = zeros(popSize,2);
        for i=1:popSize
            w = reconstruct_w(pop(i,:), M);
            F(i,:) = evaluate_objectives(w, M, k, d, theta_com, theta_scan);
        end

        % sorting
        fronts = fast_nondominated_sort(F);
        crowd = compute_crowding(F, fronts);

        % tournament selection
        mating = tournament_selection(pop, fronts, crowd);

        % SBX + mutation
        offspring = zeros(size(pop));
        for i = 1:2:popSize
            [c1, c2] = sbx_crossover(mating(i,:), mating(i+1,:), etaC, pC);
            c1 = poly_mutation(c1, etaM, pM);
            c2 = poly_mutation(c2, etaM, pM);
            offspring(i,:) = c1;
            offspring(i+1,:) = c2;
        end

        offspring = normalize_population(offspring, M);

        % combine
        combined = [pop; offspring];
        N = size(combined,1);
        F2 = zeros(N,2);
        for i=1:N
            w = reconstruct_w(combined(i,:), M);
            F2(i,:) = evaluate_objectives(w, M, k, d, theta_com, theta_scan);
        end

        fronts2 = fast_nondominated_sort(F2);
        crowd2 = compute_crowding(F2, fronts2);

        pop = environmental_selection(combined, F2, fronts2, crowd2, popSize);

        if mod(gen,10)==0
            fprintf('Generation %d / %d\n', gen, maxGen);
        end
    end

    %% ================= FINAL EVALUATION ==========================
    F_final = zeros(popSize,2);
    for i=1:popSize
        w = reconstruct_w(pop(i,:), M);
        F_final(i,:) = evaluate_objectives(w, M, k, d, theta_com, theta_scan);
    end

    %% =============== PARETO FRONT PLOT ===========================
    figure('Name','Pareto Front JCAS','NumberTitle','off');
    scatter(-F_final(:,1), F_final(:,2), 40, 'filled');
    xlabel('Communication Gain |a^H(\theta_c)w|^2');
    ylabel('Peak Sidelobe Level');
    title('NSGA-II Pareto Front for Multibeam JCAS');
    grid on;

    %% =============== GET REPRESENTATIVE SOLUTIONS =================
    [~, idx_comm_best] = max(-F_final(:,1));
    [~, idx_side_best] = min(F_final(:,2));

    mid_idx = floor(median(1:popSize));   % approximate middle solution

    w_comm = reconstruct_w(pop(idx_comm_best,:), M);
    w_side = reconstruct_w(pop(idx_side_best,:), M);
    w_mid  = reconstruct_w(pop(mid_idx,:), M);

    %% ============== BEAMPATTERN PLOTS =============================
    figure('Name','Beam Patterns','NumberTitle','off');
    hold on;

    plot_pattern(w_comm,  M, k, d, theta_full, 'Best Communication Gain');
    plot_pattern(w_side,  M, k, d, theta_full, 'Best Sidelobe Level');
    plot_pattern(w_mid,   M, k, d, theta_full, 'Middle Solution');

    xlabel('Angle (degrees)');
    ylabel('Normalized Gain (dB)');
    title('Radiation Patterns for Selected Pareto Solutions');
    legend('show');
    grid on;
end

%% ======================= OBJECTIVES ===============================
function F = evaluate_objectives(w, M, k, d, theta_com, theta_scan)

    a = @(th) exp(1j * k * d * (0:M-1)' * sind(th));

    gc = abs(a(theta_com)' * w).^2;
    F1 = -gc;

    side = abs(a(theta_scan)' * w);
    F2 = max(side);

    F = [F1, F2];
end

%% ======================= PLOTTING FUNCTION ========================
function plot_pattern(w, M, k, d, theta_range, labelname)
    a = @(th) exp(1j * k * d * (0:M-1)' * sind(th));
    vals = abs(a(theta_range)' * w).^2;
    vals_db = 10*log10(vals / max(vals));
    plot(theta_range, vals_db, 'DisplayName', labelname, 'LineWidth', 1.6);
end

%% ======================= RECONSTRUCT w ============================
function w = reconstruct_w(chrom, M)
    realp = chrom(1:M).';
    imagp = chrom(M+1:2*M).';
    w = complex(realp, imagp);
    if norm(w)==0
        w = ones(M,1)/sqrt(M);
    else
        w = w / norm(w);
    end
end

%% ======================= NORMALIZE POP ============================
function pop = normalize_population(pop, M)
    [N, ~] = size(pop);
    for i=1:N
        w = reconstruct_w(pop(i,:), M);
        pop(i,1:M) = real(w).';
        pop(i,M+1:2*M) = imag(w).';
    end
end

%% ======================= FAST NONDOM SORT ==========================
function fronts = fast_nondominated_sort(F)
    N = size(F,1);
    S = cell(N,1);
    n = zeros(N,1);
    rank = zeros(N,1);

    fronts = {};
    F1 = [];

    for p=1:N
        S{p} = [];
        n(p) = 0;
        for q=1:N
            if dominates(F(p,:), F(q,:))
                S{p}=[S{p} q];
            elseif dominates(F(q,:), F(p,:))
                n(p)=n(p)+1;
            end
        end
        if n(p)==0
            rank(p)=1;
            F1=[F1 p];
        end
    end
    fronts{1} = F1;

    i=1;
    while ~isempty(fronts{i})
        Q=[];
        for p=fronts{i}
            for q=S{p}
                n(q)=n(q)-1;
                if n(q)==0
                    rank(q)=i+1;
                    Q=[Q q];
                end
            end
        end
        i=i+1;
        fronts{i}=Q;
    end

    if isempty(fronts{end})
        fronts(end)=[];
    end
end

function b = dominates(x, y)
    b = all(x <= y) && any(x < y);
end

%% ======================= CROWDING DISTANCE =========================
function crowd = compute_crowding(F, fronts)
    N = size(F,1);
    M = size(F,2);
    crowd = zeros(N,1);

    for fi = 1:numel(fronts)
        idx = fronts{fi};
        if isempty(idx), continue; end
        Ff = F(idx,:);
        n  = length(idx);
        dist = zeros(n,1);

        for m=1:M
            [vals, order] = sort(Ff(:,m));
            dist(order(1)) = Inf;
            dist(order(end)) = Inf;

            vmax = vals(end); vmin = vals(1);
            if vmax > vmin
                for j=2:n-1
                    dist(order(j)) = dist(order(j)) + ...
                        (vals(j+1) - vals(j-1)) / (vmax - vmin);
                end
            end
        end

        for j=1:n
            crowd(idx(j)) = dist(j);
        end
    end
end

%% ======================= SELECTION ================================
function matingPool = tournament_selection(pop, fronts, crowd)
    N = size(pop,1);
    matingPool = zeros(size(pop));

    rank = zeros(N,1);
    for i=1:numel(fronts)
        rank(fronts{i}) = i;
    end

    for i=1:N
        a = randi(N); b = randi(N);
        if rank(a) < rank(b)
            w = a;
        elseif rank(b) < rank(a)
            w = b;
        else
            if crowd(a) > crowd(b)
                w = a;
            else
                w = b;
            end
        end
        matingPool(i,:) = pop(w,:);
    end
end

%% ======================= SBX CROSSOVER =============================
function [c1,c2] = sbx_crossover(p1, p2, etaC, pC)
    D = length(p1);
    c1 = p1; c2 = p2;

    if rand <= pC
        for i=1:D
            u = rand;
            if u <= 0.5
                beta = (2*u)^(1/(etaC+1));
            else
                beta = (1/(2*(1-u)))^(1/(etaC+1));
            end
            c1(i) = 0.5*((1+beta)*p1(i) + (1-beta)*p2(i));
            c2(i) = 0.5*((1-beta)*p1(i) + (1+beta)*p2(i));
        end
    end
    c1 = min(max(c1, -1), 1);
    c2 = min(max(c2, -1), 1);
end

%% ======================= POLYNOMIAL MUTATION ======================
function c = poly_mutation(c, etaM, pM)
    D = length(c);
    for i=1:D
        if rand <= pM
            u = rand;
            if u < 0.5
                delta = (2*u)^(1/(etaM+1)) - 1;
            else
                delta = 1 - (2*(1-u))^(1/(etaM+1));
            end
            c(i) = c(i) + delta;
        end
    end
    c = min(max(c, -1), 1);
end

%% ======================= ENVIRONMENTAL SELECTION ==================
function nextPop = environmental_selection(combined, F, fronts, crowd, popSize)
    nextPop = zeros(popSize, size(combined,2));
    cur = 1;

    for fi = 1:numel(fronts)
        idx = fronts{fi};
        if cur + length(idx) - 1 <= popSize
            nextPop(cur:cur+length(idx)-1, :) = combined(idx,:);
            cur = cur + length(idx);
        else
            remain = popSize - cur + 1;
            cds = crowd(idx);
            [~, order] = sort(cds, 'descend');
            selected = idx(order(1:remain));
            nextPop(cur:end, :) = combined(selected,:);
            break;
        end
    end
end