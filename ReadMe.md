# Gym Snake

A snake environment based on OpenAI gym.

## Dependencies

* [Gym](https://github.com/openai/gym)

Run it like a Gym environment:

```python
import gym
import gym_snake

env = gym.make('Snake-v0')

for episode in range(10):
    observation = env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        if done:
            break
```

## State

The state definition embeds two main objectives:
- For each action (Left - Ahead - Right), the distance between the head and the closest wall (body included);
- For each action, the number of times the action is necessary to reach the prey.

## Reward

- Each prey eaten grants the snake a quite large reward;
- At each time step, the snake gets a small negative reward (this forces the snake to head faster to the prey);
- At death time, a huge negative reward is returned.

## Demo

![alt text](https://github.com/yosinlpet/gym_snake/blob/master/demo.gif)

## Caveats

The state does not allow to avoid the winding of the snake on itself (see demo).
This is the only way the snake dies after 1000 episodes of training with DQN.
