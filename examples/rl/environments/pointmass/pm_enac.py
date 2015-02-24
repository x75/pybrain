#!/usr/bin/env python
#########################################################################
# Reinforcement Learning with different directsearch methods on the
# PointMassEnvironment 
#
# Requirements: pylab (for plotting only). If not available, comment the
# last 3 lines out
#########################################################################

__author__ = "Oswald Berthold, Thomas Rueckstiess, Frank Sehnke"

import argparse

from pybrain.tools.example_tools import ExTools
from pybrain.tools.shortcuts import buildNetwork
# from pybrain.rl.environments.cartpole import CartPoleEnvironment, BalanceTask
from pybrain.rl.environments.pointmass import PointMassEnvironment, StabilizationTask, DiscreteStabilizationTask
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import ENAC
from pybrain.rl.learners import Reinforce
from pybrain.rl.learners import SARSA
from pybrain.rl.learners import ActionValueNetwork
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.rl.agents import OptimizationAgent
from pybrain.optimization import CMAES
from pybrain.optimization import PGPE
from pybrain.optimization import FEM
from pybrain.optimization import ExactNES

import signal, rospy, sys

# learners
# mode = "enac"
# mode = "reinf" # slow
# mode = "sarsa"
# optimizers
# mode = "cma"
# mode = "fem" # nice
# mode = "pgpe" # nice
# mode = "nes" # nice
modes = ["enac", "reinf", "sarsa", "cma", "fem", "nes", "pgpe"]


def handler(signum, frame):
    print ('Signal handler called with signal', signum)
    # al.savelogs()
    # l.isrunning = False
    # if not args.batch:
    rospy.signal_shutdown("ending")
    sys.exit(0)
    # raise IOError("Couldn't open device!")

signal.signal(signal.SIGINT, handler)

def main(args):
    mode = args.mode
    
    if mode in ["enac", "reinf", "sarsa"]:
        batch=50 #number of samples per learning step
        prnts=4 #number of learning steps after results are printed
        # epis=4000/batch/prnts #number of roleouts
        epis=1000/batch/prnts #number of roleouts
        # numbExp=10 #number of experiments
        numbExp=1 #number of experiments
        kind = "learner"
    elif mode in ["cma", "fem", "nes"]:
        batch = 2
        prnts = 100
        epis = 4000/batch/prnts
        numbExp = 1
        kind = "optimizer"
    elif mode == "pgpe":
        batch = 1
        prnts = 100
        epis = 4000/batch/prnts
        numbExp = 1
        kind = "optimizer"

    et = ExTools(batch, prnts, kind = kind) # tool for printing and plotting
    print "mode", mode
    print "episodes", epis

    numActions = 21
    
    for runs in range(numbExp):
        # create environment
        env = PointMassEnvironment()
        # create task
        if mode == "sarsa":
            # discrete stabilization task
            task = DiscreteStabilizationTask(env, 1000, numActions)
            # create actionvalue network
            anet = ActionValueNetwork(4, numActions)
        else:
            # standard stabilization task
            task = StabilizationTask(env, 1000)
            # create controller network
            net = buildNetwork(4, 1, bias=True)
        # create agent with controller and learner (and its options)
        if mode == "enac":
            agent = LearningAgent(net, ENAC())
        elif mode == "reinf":
            agent = LearningAgent(net, Reinforce())
        elif mode == "sarsa":
            agent = LearningAgent(anet, SARSA())
        elif mode == "cma":
            agent = OptimizationAgent(net, CMAES(storeAllEvaluations = True))
        elif mode == "pgpe":
            agent = OptimizationAgent(net, PGPE(storeAllEvaluations = True))
        elif mode == "fem":
            agent = OptimizationAgent(net, FEM(storeAllEvaluations = True))
        elif mode == "nes":
            agent = OptimizationAgent(net, ExactNES(storeAllEvaluations = True))
            
        et.agent = agent
        # create the experiment
        experiment = EpisodicExperiment(task, agent)

        # Do the experiment
        for updates in range(epis):
            for i in range(prnts):
                experiment.doEpisodes(batch)
            if mode in ["enac", "reinf", "sarsa"]:
                state, action, reward = agent.learner.dataset.getSequence(agent.learner.dataset.getNumSequences()-1)
                et.printResults(reward.sum(), runs, updates)
            elif mode in ["cma", "pgpe", "fem", "nes"]:
                et.printResults((agent.learner._allEvaluations)[-50:-1], runs, updates)
            print "run, update", runs, updates

        et.addExps()
        print "run", runs
    et.showExps()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="Learning mode, any of " + ", ".join(modes))
    args = parser.parse_args()
    
    main(args)
