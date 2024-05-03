import numpy as np
from deap import base, creator, tools, algorithms
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import torch
import random
import copy
import warnings
import CLlosses
import DataProcessing as DP


warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='deap.*')

def GA_centroids(model, x_train, latent_size, num_classes, generations, population_size=100):
    device = next(model.parameters()).device 
    if not torch.is_tensor(x_train):
        x_train = torch.tensor(x_train, dtype=torch.float).to(device)
    else:
        x_train = x_train.to(device) 


    projections = model(x_train).detach().cpu().numpy()
    latent_space_low = np.min(projections)
    latent_space_high = np.max(projections)
    num_centroids = num_classes
    kmeans = KMeans(n_clusters=num_centroids, init='k-means++', random_state=0)
    kmeans.fit(projections)
    init_centroids = kmeans.cluster_centers_
    def fitness(individual):        
        centroids = np.array(individual).reshape((num_centroids, -1))
        distances = cdist(centroids, centroids, 'euclidean')
        np.fill_diagonal(distances, np.inf)
        min_distance = np.min(distances)
        return -min_distance,
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, latent_space_low, latent_space_high)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, num_centroids*latent_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, stats=stats, verbose=False) 
    best_individual = tools.selBest(pop, k=1)[0]
    optimized_centroids = np.array(best_individual).reshape((num_centroids, -1))
    return torch.from_numpy(optimized_centroids).float()