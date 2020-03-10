from agent import GlobalAgent
import gym


def main():
    agent = GlobalAgent('LunarLanderContinuous-v2')
    agent.train(1500)


if __name__ == "__main__":
    main()
