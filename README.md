# 2048 Reinforcement Learning

Let's play 2048 with reinforcement learning!

**Current status:** in progress


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

```python
from py2048 import Console
console = Console()
console.start()
```


## TODO

- [x] Build a 2048 game with Python
- [x] Wrap 2048 into an OpenAI Gym like environment
- [ ] Perform reinforcement learning methods on 2048