import time
import numpy as np

from pgpelib import PGPE

from sketch_env import SketchEnv
from sketch_config import config

env = SketchEnv()

env._init_all(
        config
)

num_iterations = config['num_iterations']
pop_size = config['pop_size']
n_params = env.action_dim * config['max_steps']

def run_evo(
        pgpe=None,
):

    for i in range(1, 1 + num_iterations):
        start = time.time()
        solutions = pgpe.ask()
        fitnesses = []
        
        for solution in solutions:
            fitness = set_and_get_gym(solution)
            fitnesses.append(fitness)

        pgpe.tell(fitnesses)      
        print("Iteration:", i, "  median score:", np.median(fitnesses), "; max score: ", np.max(fitnesses), " (", time.time()-start, "s)")

    return pgpe.center.copy()

def set_and_get_gym(solution):
   
    env.reset()
    rewards = []
   
    for i in range(config['max_steps']):
        start = i * env.action_dim
        end = (i + 1) * env.action_dim
        action = solution[start:end]
        _, reward, terminal, __ = env.step(action)
        rewards.append(reward)
        
        if terminal:
            env.render()
            break

    return sum(rewards)

if __name__ == '__main__':

    pgpe = PGPE(
                solution_length=n_params,
                popsize=pop_size,
                optimizer='clipup',
                optimizer_config={'max_speed': 0.15},
    )

    run_evo(
                    pgpe,
    )