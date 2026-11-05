import random
 
def fitness(chromosome, feature_scores):
    selected = [i for i, bit in enumerate(chromosome) if bit == 1]
    if not selected:
        return 0.0
    # Average relevance rewards useful features; a small penalty discourages
    # unnecessarily large feature subsets.
    relevance = sum(feature_scores[i] for i in selected) / len(selected)
    penalty = 0.001 * len(selected)
    return relevance - penalty
 
def genetic_algorithm(n_features=30, population_size=20,
                      generations=100, mutation_rate=0.02, seed=7):
    random.seed(seed)
    feature_scores = [random.random() for _ in range(n_features)]
    population = [
        [random.randint(0, 1) for _ in range(n_features)]
        for _ in range(population_size)
    ]
 
    for chromosome in population:
        if sum(chromosome) == 0:
            chromosome[random.randrange(n_features)] = 1
 
    for generation in range(generations):
        population.sort(
            key=lambda c: fitness(c, feature_scores), reverse=True
        )
        parents = population[:population_size // 2]
        offspring = []
 
        while len(offspring) < population_size:
            p1, p2 = random.sample(parents, 2)
            point = random.randint(1, n_features - 1)
            child = p1[:point] + p2[point:]
 
            for i in range(n_features):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]
 
            if sum(child) == 0:
                child[random.randrange(n_features)] = 1
            offspring.append(child)
 
        population = offspring
 
    best = max(population, key=lambda c: fitness(c, feature_scores))
    selected = [i for i, bit in enumerate(best) if bit]
    return selected, fitness(best, feature_scores)
 
if __name__ == "__main__":
    features, best_fitness = genetic_algorithm()
    print("Selected feature indices:", features)
    print("Number of available features:", 30)
    print("Number of selected features:", len(features))
    print("Best fitness:", round(best_fitness, 6))
