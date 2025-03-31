import random
import numpy as np
from vpython import vector
from vpython_3d_viewer import Point, Link, Individual, visualize_individual

POPULATION_SIZE = 1
NUM_POINTS = 5
NUM_LINKS = 5
TEST_FORCES = [vector(0, -1, 0), vector(0, 1, 0), vector(1, 0, 0), vector(-1, 0, 0)]

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]

    def union(self, p, q):
        rootP, rootQ = self.find(p), self.find(q)
        if rootP != rootQ:
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1
            return True
        return False

def grid_based_points():
    grid_size = int(NUM_POINTS ** (1/3)) + 1
    points = []
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                if len(points) < NUM_POINTS:
                    points.append(Point(x * 0.5, y * 0.5, z * 0.5))
    return points[:NUM_POINTS]

def random_individual():
    points = grid_based_points()
    links = []
    used_pairs = set()
    uf = UnionFind(NUM_POINTS)
    
    while len(links) < NUM_LINKS:
        p1, p2 = random.sample(range(NUM_POINTS), 2)
        if (p1, p2) not in used_pairs and (p2, p1) not in used_pairs and uf.union(p1, p2):
            links.append(Link(p1, p2, random.uniform(0.05, 0.2)))
            used_pairs.add((p1, p2))
    
    return Individual(points, links)

def propagate_force(individual, applied_force, start_point_index):
    force_distribution = {i: vector(0, 0, 0) for i in range(len(individual.points))}
    force_distribution[start_point_index] = applied_force
    
    queue = [start_point_index]
    while queue:
        p_a = queue.pop(0)
        force_a = force_distribution[p_a]
        
        for link in individual.links:
            if link.p1_index == p_a or link.p2_index == p_a:
                p_b = link.p2_index if link.p1_index == p_a else link.p1_index
                link_vector = individual.points[p_b].coord - individual.points[p_a].coord
                angle_factor = force_a.hat.dot(link_vector.hat)
                propagated_force = force_a.mag * angle_factor * link_vector.hat
                
                if propagated_force.mag > 1e-6:
                    force_distribution[p_b] += propagated_force
                    queue.append(p_b)
    
    return sum(f.mag for f in force_distribution.values())

def compute_fitness(individual):
    total_force = 0
    
    for link in individual.links:
        for p_a in [link.p1_index, link.p2_index]:
            for force in TEST_FORCES:
                total_force += propagate_force(individual, force, p_a)
    
    return total_force

def genetic_algorithm():
    population = [random_individual() for _ in range(POPULATION_SIZE)]
    
    for generation in range(100):
        fitness_scores = [(compute_fitness(ind), ind) for ind in population]
        fitness_scores.sort(key=lambda x: x[0])
        
        new_population = fitness_scores[:POPULATION_SIZE // 2]
        new_population = [ind for _, ind in new_population]
        
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(new_population, 2)
            new_population.append(mutate(crossover(parent1, parent2)))
        
        population = new_population
        print(f"Generation {generation}: Best Fitness {fitness_scores[0][0]}")
    
    return population[0]

def crossover(parent1, parent2):
    mid = len(parent1.points) // 2
    new_points = parent1.points[:mid] + parent2.points[mid:]
    new_links = parent1.links[:mid] + parent2.links[mid:]
    return Individual(new_points, new_links)

def mutate(individual):
    if random.random() < 0.1:
        index = random.randint(0, len(individual.points) - 1)
        individual.points[index] = Point(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
    return individual

if __name__ == "__main__":
    best_individual = genetic_algorithm()
    print("Best Individual Fitness:", compute_fitness(best_individual))
    visualize_individual(best_individual)
