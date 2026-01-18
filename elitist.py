import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION (10D)
# ==========================================
def rastrigin_function(x):
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# ==========================================
# 2. GA OPERATORS
# ==========================================
def tournament_selection(pop, fitness, k=5): 
    indices = np.random.randint(0, len(pop), k)
    best_idx = indices[np.argmin(fitness[indices])]
    return best_idx

def sbx_crossover(parent1, parent2, eta_c=15):
    child1, child2 = parent1.copy(), parent2.copy()
    n_dims = len(parent1)
    for i in range(n_dims):
        if np.random.rand() <= 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-14:
                u = np.random.rand()
                beta_q = (2*u)**(1.0/(eta_c+1)) if u<=0.5 else (1.0/(2*(1-u)))**(1.0/(eta_c+1))
                child1[i] = 0.5*((1+beta_q)*parent1[i] + (1-beta_q)*parent2[i])
                child2[i] = 0.5*((1-beta_q)*parent1[i] + (1+beta_q)*parent2[i])
    return child1, child2

def polynomial_mutation(individual, bounds, eta_m=20, mutation_rate=None):
    mutated = individual.copy()
    low, high = bounds
    if mutation_rate is None: mutation_rate = 1.0/len(individual)
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            y = mutated[i]
            delta_l = (y-low)/(high-low)
            delta_u = (high-y)/(high-low)
            u = np.random.rand()
            if u <= 0.5:
                delta = (2.0*u + (1.0-2.0*u)*(1.0-delta_l)**(eta_m+1))**(1/(eta_m+1)) - 1.0
            else:
                delta = 1.0 - (2.0*(1.0-u) + 2.0*(u-0.5)*(1.0-delta_u)**(eta_m+1))**(1/(eta_m+1))
            mutated[i] = np.clip(y + delta*(high-low), low, high)
    return mutated

def elitist_replacement(old_pop, old_fit, off_pop, off_fit):
    combined_pop = np.vstack((old_pop, off_pop))
    combined_fit = np.hstack((old_fit, off_fit))
    sorted_idx = np.argsort(combined_fit)
    N = len(old_pop)
    return combined_pop[sorted_idx[:N]], combined_fit[sorted_idx[:N]]

# ==========================================
# 3. MAIN EXECUTION (ENGLISH OUTPUT)
# ==========================================
def run_ga():
    # Parameters
    N_DIMS = 10
    BOUNDS = (-5.12, 5.12)
    POP_SIZE = 300
    N_GEN = 300
    CROSS_RATE = 0.9
    
    pop = np.random.uniform(BOUNDS[0], BOUNDS[1], (POP_SIZE, N_DIMS))
    history_best = []
    
    print(f"Running optimization on 10D Rastrigin (Pop={POP_SIZE}, Gen={N_GEN})...")
    
    for gen in range(N_GEN):
        fitness = np.array([rastrigin_function(ind) for ind in pop])
        history_best.append(np.min(fitness))
        
        offspring = []
        for _ in range(0, POP_SIZE, 2):
            p1 = pop[tournament_selection(pop, fitness, k=4)]
            p2 = pop[tournament_selection(pop, fitness, k=4)]
            c1, c2 = sbx_crossover(p1, p2) if np.random.rand() < CROSS_RATE else (p1, p2)
            offspring.append(polynomial_mutation(c1, BOUNDS))
            offspring.append(polynomial_mutation(c2, BOUNDS))
            
        off_pop = np.array(offspring)
        off_fit = np.array([rastrigin_function(ind) for ind in off_pop])
        pop, fitness = elitist_replacement(pop, fitness, off_pop, off_fit)
        
        if gen % 50 == 0:
            print(f"Gen {gen}: Fitness = {history_best[-1]:.6f}")

    # --- FINAL RESULTS (ENGLISH) ---
    best_ind = pop[0]
    print("\n" + "="*30)
    print(f"FINAL RESULTS (10D):")
    print(f"Fitness: {history_best[-1]:.10f}")
    print(f"Solution x (rounded): \n{np.round(best_ind, 4)}")
    print("="*30)

    # Plotting (English Labels)
    plt.figure(figsize=(10, 6))
    plt.plot(history_best, 'b-', linewidth=2)
    plt.yscale('log')
    plt.title(f'Convergence Speed on 10D Rastrigin\nPop: {POP_SIZE}, Gen: {N_GEN}')
    plt.ylabel('Fitness (Log Scale)')
    plt.xlabel('Generations')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Save Image
    plt.savefig('convergence_10d.png', dpi=300, bbox_inches='tight')
    print("Graph saved to 'convergence_10d.png'")
    plt.show()

if __name__ == "__main__":
    run_ga()
