#!/usr/bin/env python
__author__ = 'Oswald Berthold, bertolos@informatik.hu-berlin.de'

from pybrain.rl.environments.pointmass import PointMassEnvironment, DiscreteStabilizationTask, StabilizationTask
from pybrain.rl.learners.valuebased import NFQ, ActionValueNetwork
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.rl.explorers import EpsilonGreedyExplorer, NormalExplorer

# from pybrain.tools.shortcuts import buildNetwork
# from pybrain.rl.learners import ENAC
# from pybrain.rl.agents import OptimizationAgent
# from pybrain.optimization import PGPE

import numpy as np
from numpy import array, arange, meshgrid, pi, zeros, mean
from matplotlib import pyplot as plt

import rospy
import sys, signal

len_episode = 1000
# len_episode = 400
render = False

numActions = 3

env = PointMassEnvironment(len_episode=len_episode)

# print dir(env)

module = ActionValueNetwork(3, numActions)
# net = buildNetwork(3, 1, bias=True)

task = DiscreteStabilizationTask(env, maxsteps=len_episode)

explorer = EpsilonGreedyExplorer()
# NFQ is continuous states, discrete actions, so nocontinuous explorer
# explorer = NormalExplorer(dim = 1, sigma=0.5)

learner = NFQ()
learner._setExplorer(explorer)
learner.explorer.epsilon = 0.4
agent = LearningAgent(module, learner)
testagent = LearningAgent(module, None)

# agent = LearningAgent(net, ENAC())
# agent = OptimizationAgent(net, PGPE(storeAllEvaluations = True))

experiment = EpisodicExperiment(task, agent)

def plotPerformance(values, fig):
    plt.figure(fig.number)
    plt.clf()
    plt.plot(values, 'o-')
    plt.gcf().canvas.draw()
    # Without the next line, the pyplot plot won't actually show up.
    plt.pause(0.001)

def plotState(x, v, a, u, fig):
    plt.figure(fig.number)
    plt.clf()
    plt.title("pos and vel")
    plt.subplot(311)
    plt.plot(x)
    plt.plot(v)
    plt.subplot(312)
    plt.title("acceleration")
    plt.plot(a)
    plt.subplot(313)
    plt.title("action")
    plt.plot(u)
    plt.gcf().canvas.draw()
    plt.pause(0.001)
    # plt.show()
    
    
def handler(signum, frame):
    print ('Signal handler called with signal', signum)
    # al.savelogs()
    # l.isrunning = False
    # if not args.batch:
    rospy.signal_shutdown("ending")
    sys.exit(0)
    # raise IOError("Couldn't open device!")

signal.signal(signal.SIGINT, handler)
    
performance = []

if not render:
    pf_fig = plt.figure()
    st_fig = plt.figure()

# # ROS init
# rospy.init_node("pm")

while(True):
	# one learning step after one episode of world-interaction
    print "experiment.doEpisodes(1)"
    # batch = 10
    experiment.doEpisodes(1)
    if not render:
        plotState(experiment.task.env.ip2d.x,
                  experiment.task.env.ip2d.v,
                  experiment.task.env.ip2d.a,
                  experiment.task.env.ip2d.u,
                  st_fig)
    # state, action, reward = agent.learner.dataset.getSequence(agent.learner.dataset.getNumSequences()-1)
    # r = np.mean(reward)
    # print "exp.task.env", experiment.task.env.ip2d
    agent.learn(1)

    # test performance (these real-world experiences are not used for training)
    if render:
        env.delay = True
    experiment.agent = testagent
    r = mean([sum(x) for x in experiment.doEpisodes(5)])
    env.delay = False
    testagent.reset()
    experiment.agent = agent

    # print "reward", r
    performance.append(r)
    if not render:
        plotPerformance(performance, pf_fig)

    print("reward avg", r)
    print("explorer epsilon", learner.explorer.epsilon)
    print("num episodes", agent.history.getNumSequences())
    print("update step", len(performance))
