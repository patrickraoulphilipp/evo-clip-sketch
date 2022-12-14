Sketching with Evolutionary Algorithms and CLIP
===
Implementation of *Evoltionary CLIP Sketches*, a gym-like environment for painting simple sketches and integration of an evolutionary algorithm to search a good solution for an input text. The repository uses OpenAI's CLIP (https://github.com/openai/CLIP) to parse the input text and calculate a reward/fitness for the gym environment and evolutionary algorithm.

The work is inspired by Google Brain's *Modern Evolution Strategies for Creativity: Fitting Concrete Images and Abstract Concepts* (https://arxiv.org/pdf/2109.08857.pdf & https://github.com/google/brain-tokyo-workshop/tree/master/es-clip), which investigates an bstract image generation process. This repository provides an easy-to-use and challenging image generation environment which can be efficiently run on CPUs.

The available evolutionary algorithm for sketch finding is PGPE, a derivative-free policy gradient estimation algorithm (https://github.com/nnaisense/pgpelib). The algorithm works sufficiently better for this environment compared to standard evolutionary strategies (at least for my initial experiments), but I will add the code and configuration parameter to run these as well. The idea to test PGPE is also inspired by Google Brain's mentioned work.

Instructions
-------------
1.  Clone this repo.

```
git clone https://github.com/patrickraoulphilipp/evo-clip-sketch
cd evo-clip-sketch
```

2. (optional) Create a virtualenv. The implementation has been tested for Python 3.9.

```
virtualenv venv
source venv/bin/activate
```

3. Install all dependencies. You need both pgelib and CLIP, which will be automatically installed from their respective git repos.

```
pip install -r requirements.txt .
```

5. Set parameters in config.py. For now, all relevant parameters are to be set in as dictionary. The most important ones to get started are **target_words** to set the textual goal for the sketch and **max_steps** to set the number of relative points to be selected. **target_dir** enables to set the target directory to save the best sketches in the process.

```python
config = {
    'target_words' : 'a cat', # text for calculating CLIP-based fitness
    'max_steps' : 20, # number of relative points for one episode/sktech
    'target_dir' : '/PATH/TO/FOLDER/', # folder for saving generated sketches
    ...
}
```

6. Run main.py to start the search process. You will a continuous stream of sketches that are being testes by the evolutionary algorithm. Only the sketches which improve the best global reward/fitness are saved.

```
python main.py
```