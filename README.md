# reinforcement-learning-super-mario-A3C
Learning to play supermario using A3C Reinforcement Learning algorithm  
Original A3C paper: https://arxiv.org/pdf/1602.01783.pdf  

## How to run
First install [Super Mario Gym](https://github.com/ppaquette/gym-super-mario)  

Then edit settings in `main.py` and `agent.py` and simply run  
```
python main.py
```

## Tip

Things to change in the super-mario-bros.lua file to send slightly less data through the pipe:
- Row 48: skip\_frames=6
- Row 445: for y=32,206 do 

## Command to kill all emulators at once  
ps | grep fceux | awk '{print $1}' | xargs kill -9
