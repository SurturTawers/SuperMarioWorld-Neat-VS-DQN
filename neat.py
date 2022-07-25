import retro
import numpy as np
#pip install opencv-python to install cv2
import cv2
import neat
import pickle

env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2')
imgarray = []

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
    ob = env.reset()
    ac = env.action_space.sample() # action with generic sample
    iny, inx, inc = env.observation_space.shape
    inx = int(inx / 8)
    iny = int(iny / 8)
    net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    current_max_fitness = 0
    fitness_current = 0
    frame = 0
    counter = 0
    score = 0
    scoreTracker = 0
    coins = 0
    coinsTracker = 0
    yoshiCoins = 0
    yoshiCoinsTracker = 0
    xPosPrevious = 0
    yPosPrevious = 0
    checkpoint = False
    checkpointValue = 0
    endOfLevel = 0
    powerUps = 0
    powerUpsLast = 0
    jump = 0
    done = False
    while not done:
        env.render()
        frame += 1
        ob = cv2.resize(ob, (inx, iny))
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = np.reshape(ob, (inx, iny))
        imgarray = ob.flatten()
        nnOutput = net.activate(imgarray)

        ob, rew, done, info = env.step(nnOutput)

        score = info['score']
        coins = info['coins']
        yoshiCoins = info['yoshiCoins']
        dead = info['dead']
        xPos = info['x']
        yPos = info['y']
        jump = info['jump']
        checkpointValue = info['checkpoint']
        endOfLevel = info['endOfLevel']
        powerUps = info['powerups']
        if score > 0:
            if score > scoreTracker:
                fitness_current = (score * 10)
                scoreTracker = score

        if coins > 0:
            if coins > coinsTracker:
                fitness_current += (coins - coinsTracker)
                coinsTracker = coins

        if yoshiCoins > 0:
            if yoshiCoins > yoshiCoinsTracker:
                fitness_current += (yoshiCoins - yoshiCoinsTracker) * 10
                yoshiCoinsTracker = yoshiCoins
        if xPos > xPosPrevious:
            if jump > 0:
                fitness_current += 10
            fitness_current += (xPos / 100)
            xPosPrevious = xPos
            counter = 0
        else:
            counter += 1
            fitness_current -= 0.1

        if yPos < yPosPrevious:
            fitness_current += 10
            yPosPrevious = yPos
        elif yPos < yPosPrevious:
            yPosPrevious = yPos

        if powerUps == 0:
            if powerUpsLast == 1:
                fitness_current -= 500
                print("Lost Upgrade")
        elif powerUps == 1:
            if powerUpsLast == 1 or powerUpsLast == 0:
                fitness_current += 0.025
            elif powerUpsLast == 2:
                fitness_current -= 500
                print("Lost Upgrade")
        powerUpsLast = powerUps

        if checkpointValue == 1 and checkpoint == False:
            fitness_current += 20000
            checkpoint = True

        if endOfLevel == 1:
            fitness_current += 1000000
            done = True

        if counter == 1000:
            fitness_current -= 125
            done = True

        if dead == 0:
            fitness_current -= 100
            done = True

        if done == True:
             print(genome_id, fitness_current)
        genome.fitness = fitness_current

def replay_genome(config_path="config-feedforward-mw",genome_path="winner.pkl"):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            config_path)
    with open('winner.pkl', "rb") as rep:
        genome = pickle.load(rep)

    genomes = [(1, genome)]
    eval_genomes(genomes, config)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        'config-feedforward-mw')
p = neat.Population(config)
#Descomentar para cargar un checkpoint de NEAT
#p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-10")
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))
#Descomentar esta linea para volver a ejecutar un ganador
#para eso comentar las lineas siguientes
#replay_genome()

#Descomentar estas lineas para ejecución normal y comentar la linea anterior
#winner = p.run(eval_genomes)
#with open('winner.pkl', 'wb') as output:
# pickle.dump(winner, output, 1)
