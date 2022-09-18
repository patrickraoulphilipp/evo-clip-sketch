config = {
    'target_words' : 'a cat', # text for calculating CLIP-based fitness
    'max_steps' : 20, # number of relative points for one episode/sktech
    'target_shape' : (200,200,3), # size of sketch image
    'target_thickness' : 3, # thickness of lines
    'point_reduction_factor' : 4,  # reduce pont distances
    'target_dir' : None, # folder for saving generated sketches
    'num_actions' : 2, # must stay =2 for choosing x,y coordinates
    'pop_size' : 100, # size of iniital sketch population
    'num_iterations' : 10 ** 6, # number of iterations to mutate population
    'clip_model' : "RN50x4" # see clip.clip._MODELS for supported models, e.g., ViT-B/32, RN50x4
}