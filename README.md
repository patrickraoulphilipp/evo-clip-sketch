Sketching with Evolutionary Algorithms and CLIP
===
Implementation of *Evoltionary CLIP Sketches*, a gym-like environment for painting simple sketches and integration of an evolutionary algorithm to search a good solution for an input text. The repository uses OpenAI's CLIP (https://github.com/openai/CLIP) to parse the input text and calculate a reward/fitness for the gym environment and evolutionary algorithm.

The work is related and inspired by Google Brain's *Modern Evolution Strategies for Creativity: Fitting Concrete Images and Abstract Concepts* (https://arxiv.org/pdf/2109.08857.pdf & https://github.com/google/brain-tokyo-workshop/tree/master/es-clip), and provides an easy to use and less costly image generation process, which is interesting to investigate.

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

5. Set parameters in sketch_config.py. For now, all relevant parameters are to be set in as dictionary. The most important ones to get started are target_words to set the textual goal for the sketch and max_steps to set the number of relative points to be selected. target_dir enables to set the target directory to save the best sketches in the process.

```
config = {
    'target_words' : ['a kitten'],
    'max_steps' : 20,
    'target_dir' : '/PATH/TO/FOLDER/',
    ...
}
```

6. Run evo.py to start the search process. You will a continuous stream of sketches that are being testes by the evolutionary algorithm. Only the sketches which improve the best global reward/fitness are saved.

```
python evo.py
```