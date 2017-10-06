from dummy_agent import Agent
from threading import Thread

def main():
    """ the main function """
    LEVEL_NAME = 'SuperMarioBros-1-1-v0'
    NUMBER_OF_AGENTS = 1
    EPISODES = 5

    """ create an array of agents"""
    agents = []
    for i in range(0,NUMBER_OF_AGENTS):
        agents.append(Agent(LEVEL_NAME))

    """ run all agents in separate threads """
    agent_threads = []

    for agent in agents:
        f = lambda: agent.train(EPISODES)
        t = Thread(target=f)
        t.start()
        agent_threads.append(t)



if __name__ == '__main__':
    main()
