#!/usr/bin/env python

import argparse, rospy, sys, signal
from pybrain.rl.environments.arm import ArmEnvironment, CartesianTask#, StabilizationTaskVel
from pybrain.rl.agents import LearningAgent
from pybrain.rl.explorers import NormalExplorer2
from pybrain.rl.learners import CACLA
from pybrain.rl.learners.valuebased import ActionValueNetwork
from pybrain.rl.experiments import ContinuousExperiment
from pybrain.structure.modules.aclayer import ACLayer, ACLayerMLP

def handler(signum, frame):
    print ('Signal handler called with signal', signum)
    rospy.signal_shutdown("ending")
    sys.exit(0)

def main(args):
    # install stop handler
    signal.signal(signal.SIGINT, handler)

    num_episodes = 10
    len_episode = 100000

    indim = 4  # sensor dim
    outdim = 3 # motor dimx
    

    # environment
    env = ArmEnvironment(len_episode=len_episode)
    # combined learning module
    module = ACLayer(indim = indim, outdim = outdim, hdim = 500)
    # module = ACLayerMLP(indim = indim, outdim = outdim, hdim = 10)
    # task
    task = CartesianTask(env, maxsteps=len_episode)
    # explorer
    explorer = NormalExplorer2(dim = outdim, sigma = 1e-3) # 1e-1
    # learner
    alpha = 2e-3
    beta  = 7e-1
    # alpha = 1e-3
    # beta  = 1e-2
    gamma = 9.99e-1
    # gamma = 5e-2
    # gamma = 1e-1
    learner = CACLA(module, task, alpha=alpha, beta=beta, gamma=gamma, explorer=explorer)
    # learner.explorer = explorer
    # agent
    agent = LearningAgent(module, learner)
    # experiment
    experiment = ContinuousExperiment(task, agent)
    print "Cacla learning"
    # experiment.doEpisodes(num_episodes)
    for episode in range(num_episodes):
        print "pm_cacla: episode", episode
        # for i in range(len_episode):
        r = experiment.doInteractionsAndLearn(len_episode)
        # print "pm_cacla: r", r
        # print i
        env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    
    main(args)
