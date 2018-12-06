# Gym Snake

A snake environment based on OpenAI gym.

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
