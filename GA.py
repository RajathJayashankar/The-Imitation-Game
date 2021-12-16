from deap import base, creator, tools, algorithms

population_size = 4
num_generations = 4
gene_length = 10


def initalize():
    # Placeholder for every individual, this is intialized as a dictionary
    population = {}
    # Population size
    npop = 20
    varmin = -10
    varmax = 10
    costfunc = manhattan_dist()
    # Each inidivdual has position(chromosomes) and heuristic cost
    for i in range(npop):
        population[i] = {'position': None, 'cost': None}
    for i in range(npop):
        population[i]['position'] = np.random.uniform(varmin, varmax, num_var)
    population[i]['cost'] = costfunc(population[i]['position'])   
    

def roulette_wheel_selection(p): #Implementaion fo roulette wheel selection
   c = np.cumsum(p) # We calculate each parent’s probability’s cumulative sum, multiply its sum with a randomly generated number.
   r = sum(p) * np.random.rand()
   ind = np.argwhere(r <= c) # Returns an array of Trues and Falses based on the expression passed as a parameter.
   return ind[0][0]


def crossover(p1, p2): #Input the populations
   c1 = copy.deepcopy(p1)
   c2 = copy.deepcopy(p2)
   alpha = np.random.uniform(0, 1, *(c1['position'].shape)) # We take the complement of alpha values to produce offspring-2
   c1['position'] = alpha*p1['position'] + (1-alpha)*p2['position']
   c2['position'] = alpha*p2['position'] + (1-alpha)*p1['position'] 
   return c1, c2


def mutate(c, mu, sigma): # mu - mutation rate. % of gene to be modified, sigma - step size of mutation
   y = copy.deepcopy(c)     
   flag = np.random.rand(*(c['position'].shape)) <= mu  # array of True and Flase, indicating the mutation position
   ind = np.argwhere(flag)
   y['position'][ind] += sigma * np.random.randn(*ind.shape)
   return y


# Evaluate first off spring
# calculate cost function of child 1
c1['cost'] = costfunc(c1['position'])
if type(bestsol_cost) == float:
  # replacing best solution in every generation/iteration
  if c1['cost'] < bestsol_cost:
    bestsol_cost = copy.deepcopy(c1)
else:
   # replacing best solution in every generation/iteration
   if c1['cost'] < bestsol_cost['cost']:
     bestsol_cost = copy.deepcopy(c1)
# Evaluate second off spring
if c2['cost'] < bestsol_cost['cost']:
bestsol_cost = copy.deepcopy(c2)

beta = 1
for i in range(len(population)):
   # list of all the population cost
   costs.append(population[i]['cost'])
costs = np.array(costs)
avg_cost = np.mean(costs)
if avg_cost != 0:
   costs = costs/avg_cost
probs = np.exp(-beta*costs)

creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
creator.create('Individual', list , fitness = creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n = gene_length)
toolbox.register('population', tools.initRepeat, list , toolbox.individual)

toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
toolbox.register('select', tools.selRoulette)
toolbox.register('evaluate', train_evaluate)

population = toolbox.population(n = population_size)
r = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1, ngen = num_generations, verbose = False)

def small_unet():
    inputs = Input((128, 128, 1))


    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model
small_unet().summary()