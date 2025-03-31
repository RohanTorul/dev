import random
import numpy as np
from matplotlib import pyplot as plt
from vpython import vector
from vpython_3d_viewer import Point, Link, Individual, visualize_individual,visualize_individuals
from math import sin, cos, tan,pi
from itertools import combinations

from joblib import Parallel, delayed


POPULATION_SIZE = 100
NUM_POINTS = 20
NUM_LINKS = 16
TEST_FORCES = [vector(0, -1, 0), vector(0, 1, 0), vector(1, 0, 0), vector(-1, 0, 0)]
MAX_ITERATION = 100
SMOOTHING = [0.5,100]

def compute_fitness_parallel(individual):
    return compute_fitness(individual)*SMOOTHING[0] + SMOOTHING[1], individual

def evaluate_population_fitness(population):
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()//2) as pool:
    #     fitness_scores = pool.map(compute_fitness_parallel, population)
    # return sorted(fitness_scores, key=lambda x: x[0])  # Sort by fitness
    fitness_scores = Parallel(n_jobs=-2)(delayed(compute_fitness_parallel)(ind) for ind in population)
    return sorted(fitness_scores, key=lambda x: x[0])

def blend_crossover(parent1, parent2):
    mid = len(parent1.points) // 2
    alpha = random.uniform(-0.1, 1.1)  # Allows some extrapolation
    new_points = [Point(alpha * p1.coord.x + (1 - alpha) * p2.coord.x,
                        alpha * p1.coord.y + (1 - alpha) * p2.coord.y ,
                        alpha * p1.coord.z + (1 - alpha) * p2.coord.z
                        )for p1, p2 in zip(parent1.points, parent2.points)]
    return Individual(new_points,parent1.links[:mid] + parent2.links[mid:])


def roulette_wheel_selection(population):
    total_fitness = sum(1 / (compute_fitness(ind) + 1e-6) for ind in population)
    pick = random.uniform(0, total_fitness)
    current = 0
    for ind in population:
        current += 1 / (compute_fitness(ind) + 1e-6)
        if current > pick:
            return ind


def tournament_selection(population, k=3):
    """ Selects a parent using tournament selection. """
    candidates = random.sample(population, k)
    return min(candidates, key=lambda ind: compute_fitness(ind))  # Lower fitness is better

def dynamic_mutation(individual, generation, max_generations):
    """ Adjusts mutation rate based on generation progress and ensures a valid individual. """
    mutation_rate = max(0.05, 0.2 * (1 - generation / max_generations))  # High at start, low at end
    
    if random.random() < mutation_rate:
        index = random.randint(0, len(individual.points) - 1)
        new_point = Point(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        individual.points[index] = new_point
    
    return Individual(individual.points, individual.links)  # Ensure it returns a valid Individual



def grid_based_points():
    grid_size = int(NUM_POINTS ** (1/3)) + 1
    points = []
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                if len(points) < NUM_POINTS:
                    points.append(Point(x * 0.5, y * 0.5, z * 0.5))
    return points[:NUM_POINTS]

def ensure_connectivity(points, links):
    adjacency_list = {i: set() for i in range(len(points))}
    for link in links:
        adjacency_list[link.p1_index].add(link.p2_index)
        adjacency_list[link.p2_index].add(link.p1_index)
    
    visited = set()
    queue = [0]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            queue.extend(adjacency_list[node] - visited)
    
    return (len(visited) / len(points))*100

def random_individual():
   
    points = grid_based_points()#[Point(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(NUM_POINTS)]
    links = []
    used_pairs = set()
    
    max_tries = list(combinations(range(NUM_POINTS), 2))
    while len(links) < NUM_LINKS:
        p1, p2 = random.choice(max_tries)
        if (p1, p2) not in used_pairs and (p2, p1) not in used_pairs:
            links.append(Link(p1, p2, 0.05))
            used_pairs.add((p1, p2))
            max_tries.remove((p1,p2))
    
       
                
        # while len(links) < NUM_LINKS and tries < max_tries:
        #     tries += 1
        #     p1, p2 = random.sample(range(NUM_POINTS), 2)
        #     if (p1, p2) not in used_pairs and (p2, p1) not in used_pairs and (points[p1].coord.hat.dot(points[p2].coord.hat))/(points[p1].coord.mag * points[p2].coord.mag) < cos(pi) and (points[p1].coord.hat.dot(points[p2].coord.hat))/(points[p1].coord.mag * points[p2].coord.mag) >= cos(pi/8) :
        #         links.append(Link(p1, p2, random.uniform(0.05, 0.2)))
        #         used_pairs.add((p1, p2))
        
    return Individual(points, links)

def propagate_force(individual, applied_force, start_point_index, start_link):
    force_distribution = {i: vector(0, 0, 0) for i in range(len(individual.points))}
    force_distribution[start_point_index] = applied_force
    
    queue = [(start_point_index, start_link)]
    visited_links = set()

    while queue:
        p_a, prev_link = queue.pop(0)
        force_a = force_distribution[p_a]
        
        for link in individual.links:
            if link in visited_links or link is prev_link:
                continue  # Skip already visited links
            
            if (link.p1_index == p_a or link.p2_index == p_a)and link is not start_link:
                p_b = link.p2_index if link.p1_index == p_a else link.p1_index
                link_vector = individual.points[p_b].coord - individual.points[p_a].coord
                angle_factor = force_a.hat.dot(link_vector.hat)
                propagated_force = force_a.mag * angle_factor * link_vector.hat
                
                if 1e-6 < propagated_force.mag < applied_force.mag:
                    force_distribution[p_b] += propagated_force
                    queue.append((p_b, link))
                    visited_links.add(link)  # Mark this link as visited
    
    return sum(f.mag for f in force_distribution.values())


def compute_fitness(individual):
    total_force = 0
    Bonus_fitness = 0
    Bonus_fitness = ensure_connectivity(individual.points, individual.links)
    #print("Bonus Fitness: ", Bonus_fitness)
    for link in individual.links:
        for p_a in [link.p1_index, link.p2_index]:
            for force in TEST_FORCES:
                total_force += propagate_force(individual, force, p_a,link)
    
    return total_force - Bonus_fitness

def genetic_algorithm():
    population = [random_individual() for _ in range(POPULATION_SIZE)]
    print("Initialisation complete")
    Most_fits = []
    all_Max_fitnesses = [] 
    for generation in range(MAX_ITERATION):
        fitness_scores = evaluate_population_fitness(population)#[(compute_fitness(ind)*SMOOTHING[0] + SMOOTHING[1], ind) for ind in population]
        fitness_scores.sort(key=lambda x: x[0])
        
        new_population = fitness_scores[:POPULATION_SIZE // 2]
        new_population = [ind for _, ind in new_population]
        
        while len(new_population) < POPULATION_SIZE:
            parent1 = roulette_wheel_selection(population)
            parent2 = roulette_wheel_selection(population)
            offspring = crossover(parent1, parent2)
            offspring = dynamic_mutation(offspring, generation, MAX_ITERATION)  # Ensure a valid return
            new_population.append(offspring)

            offspring = crossover(parent2, parent1)
            offspring = dynamic_mutation(offspring, generation, MAX_ITERATION)  # Ensure a valid return
            new_population.append(offspring)

        population = new_population
        Most_fits.append(population[0])
        all_Max_fitnesses.append(fitness_scores[0][0])
        print(f"Generation {generation}: Best Fitness {fitness_scores[0][0]}")
    
    return (population[0],Most_fits,all_Max_fitnesses)

def crossover(parent1, parent2):
    mid = len(parent1.points) // 2
    new_points = parent1.points[:mid] + parent2.points[mid:]
    new_links = parent1.links[:mid] + parent2.links[mid:]
    return Individual(new_points, new_links)

def mutate(individual):
    if random.random() < 0.5:
        index = random.randint(0, len(individual.points) - 1)
        individual.points[index] = Point(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
    return individual

if __name__ == "__main__":
    best_individual, bests_per_gen, all_max_fitnesses = genetic_algorithm()
    generations = np.arange(MAX_ITERATION)
    plt.plot(generations,all_max_fitnesses)
    plt.show()
    print("Best Individual Fitness:", min(all_max_fitnesses))
    # for ind_id in range(len(bests_per_gen)):
    #     print("ind = ", ind_id)
    #     for p in bests_per_gen[ind_id].points:
    #         p.coord += vector(2*ind_id,0,0)
    #visualize_individual(best_individual)
    visualize_individuals(bests_per_gen)
