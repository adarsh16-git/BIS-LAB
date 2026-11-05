import random
 
def objective(position):
    # Demonstration objective: maximize the number of selected
    # relevant features while discouraging overly large subsets.
    relevant = {2, 5, 8, 11, 14, 17}
    selected = {i for i, value in enumerate(position) if value >= 0.5}
    if not selected:
        return 0.0
    score = len(selected & relevant) / len(relevant)
    penalty = 0.01 * max(0, len(selected) - len(relevant))
    return score - penalty
 
def particle_swarm(n_features=30, n_particles=20,
                   iterations=20, seed=10):
    random.seed(seed)
    particles = [
        [random.random() for _ in range(n_features)]
        for _ in range(n_particles)
    ]
    velocities = [
        [random.uniform(-0.1, 0.1) for _ in range(n_features)]
        for _ in range(n_particles)
    ]
 
    pbest = [p[:] for p in particles]
    pbest_score = [objective(p) for p in particles]
    g_index = max(range(n_particles), key=lambda i: pbest_score[i])
    gbest = pbest[g_index][:]
    gbest_score = pbest_score[g_index]
 
    w, c1, c2 = 0.7, 1.5, 1.5
 
    for _ in range(iterations):
        for i in range(n_particles):
            for d in range(n_features):
                r1, r2 = random.random(), random.random()
                velocities[i][d] = (
                    w * velocities[i][d]
                    + c1 * r1 * (pbest[i][d] - particles[i][d])
                    + c2 * r2 * (gbest[d] - particles[i][d])
                )
                particles[i][d] += velocities[i][d]
                particles[i][d] = max(0.0, min(1.0, particles[i][d]))
 
            score = objective(particles[i])
            if score > pbest_score[i]:
                pbest[i] = particles[i][:]
                pbest_score[i] = score
                if score > gbest_score:
                    gbest = particles[i][:]
                    gbest_score = score
 
    chosen = [i for i, value in enumerate(gbest) if value >= 0.5]
    return chosen, gbest_score
 
if __name__ == "__main__":
    selected, score = particle_swarm()
    print("Iterations performed:", 20)
    print("Best fitness:", round(score, 4))
    print("Selected features:", len(selected))
    print("Feature indices:", selected)
