from agent import Agent
import gym


def main():
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(env)
    agent.train(1500)


if __name__ == "__main__":
    main()
