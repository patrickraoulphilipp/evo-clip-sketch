import time
import numpy as np

from pgpelib import PGPE

from evo_clip_sketch.sketch_env import SketchEnv

def run_evo(
        config=None,
):

    env = SketchEnv()
    env._init_all(
        config
    )

    num_iterations = config['num_iterations']
    pop_size = config['pop_size']
    n_params = env.action_dim * config['max_steps']

    pgpe = PGPE(
                solution_length=n_params,
                popsize=pop_size,
                optimizer='clipup',
                optimizer_config={'max_speed': 0.15},
    )

    for i in range(1, 1 + num_iterations):
        start = time.time()
        solutions = pgpe.ask()
        fitnesses = []
        min_stats = []
        max_stats = []
        
        for solution in solutions:
            fitness = get_fitness_from_gym(
                            env=env,
                            solution=solution,
                            config=config,
                       )
            fitnesses.append(fitness)
            min_stats.append(np.min(solution))
            max_stats.append(np.max(solution))

        pgpe.tell(fitnesses)      
        print(
                "Iteration:", i, 
                "  median score:", np.median(fitnesses), 
                "; max score: ", np.max(fitnesses), 
                " (", time.time()-start, "s)",
                "; min value: ", np.min(min_stats), 
                "; max_value: ", np.max(max_stats)
        )

def get_fitness_from_gym(
        env=None,
        solution=None,
        config=None
):
   
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