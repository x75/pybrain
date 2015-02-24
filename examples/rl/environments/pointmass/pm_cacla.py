#!/usr/bin/env python
############################################################
# Reinforcement Learning with the Cacla algorithm on the
# PointMassEnvironment
############################################################

__author__ = "Oswald Berthold"

from pybrain.tools.example_tools import ExTools
from pybrain.tools.shortcuts import buildNetwork
# from pybrain.rl.environments.cartpole import CartPoleEnvironment, BalanceTask
from pybrain.rl.environments.pointmass import PointMassEnvironment, StabilizationTask, StabilizationTaskVel
from pybrain.rl.agents import LearningAgent
# from pybrain.rl.learners import ENAC
# from pybrain.rl.learners import Reinforce
from pybrain.rl.explorers import NormalExplorer, NormalExplorer2
from pybrain.rl.learners import CACLA
from pybrain.rl.learners.valuebased import ActionValueNetwork
from pybrain.rl.experiments import ContinuousExperiment
from pybrain.structure.modules.aclayer import ACLayer
# from pybrain.rl.agents import OptimizationAgent
#from pybrain.optimization import CMAES
#from pybrain.optimization import PGPE
#from pybrain.optimization import FEM
#from pybrain.optimization import ExactNES

import signal, rospy, sys

import numpy as np
# np.random.seed(1244)
# np.random.seed(1247)
# np.random.seed(1256)

# et = ExTools(batch, prnts, kind = kind) # tool for printing and plotting

def handler(signum, frame):
    print ('Signal handler called with signal', signum)
    # al.savelogs()
    # l.isrunning = False
    # if not args.batch:
    rospy.signal_shutdown("ending")
    sys.exit(0)
    # raise IOError("Couldn't open device!")

def main(args):
    # install stop handler
    signal.signal(signal.SIGINT, handler)

    num_episodes = 10
    len_episode = 60000

    # environment
    env = PointMassEnvironment(len_episode=len_episode+1)
    # controllerA
    # action = buildNetwork(3, 1, bias=True)
    # print "action", action
    # print "dir(action)", dir(action)
    # controllerV
    # value = buildNetwork(3, 1, bias=True)
    # value = ActionValueNetwork(3, 1)
    # print "dir(value)", dir(value)
    # sys.exit()
    # combined learning module
    module = ACLayer(indim = 3, outdim = 1, hdim = 200)
    # task
    task = StabilizationTaskVel(env, maxsteps=len_episode)
    # explorer
    explorer = NormalExplorer2(dim = 1, sigma = 5e-2) # 1e-1
    # learner
    # alpha = 1e-2
    # beta  = 1e-1
    alpha = 1e-3
    beta  = 1e-2
    gamma = 9.9e-1
    learner = CACLA(module, task, alpha=alpha, beta=beta, gamma=gamma)
    learner.explorer = explorer
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

if __name__ == "__main__":
    main(None)
