# reinforcement-learning-super-mario-A3C
## Under construction
Learning to play supermario using A3C algorithm

Todo:
- [X] Implement Tensorboard
- [X] Save model
- [X] crop image
- [ ] Train on Pong
- [ ] Train on tiles
- [X] Reward engineering
- [X] Timeout restart

Things to change in the super-mario-bros.lua file to send slightly less data through the pipe:
- Row 48: skip\_frames=6
- Row 445: for y=32,206 do 

## Command to kill all emulators at once  
ps | grep fceux | awk '{print $1}' | xargs kill -9
