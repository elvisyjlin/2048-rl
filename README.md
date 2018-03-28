# 2048 Reinforcement Learning

Let's play 2048 with reinforcement learning!

**Current status:** not better than the random agent


## Requirements

Install Python 3 and all its dependencies:

```bash
pip3 install -t requirements.txt
```

1. numpy
2. tensorflow
3. keras
4. h5py


## Train and Test

```bash
python3 --agent=[TYPE] --model_path=[MODEL] --train --test
```



## Environment

```python
from gym2048 import Gym2048
gym = Gym2048()
env = gym.make()

env.reset()
while True:
    action = env.action_space.sample()
    obervation, reward, done, info = env.step(action)
    env.render()
    if done: break
```


## Play in Console

You can play 2048 with arrow keys:

```bash
python3 play.py
```

```python
from py2048 import Console
console = Console()
console.start()
```


## TODO

- [x] Build a 2048 game with Python
- [x] Wrap 2048 into an OpenAI Gym like environment
- [x] Perform reinforcement learning methods on 2048
- [ ] Improve the existing RL methods
- [ ] Analyse the training progress of each method