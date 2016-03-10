# import all client and server scripts
import NERO.module as module
import NERO.client
import NERO.constants as constants
import common
import common.menu_utils
import OpenNero
import os

mod = None
step = 0


def ModMain(mode=""):
    global mod
    NERO.client.ClientMain()
    mod = NERO.module.getMod()
    mod.change_flag([constants.XDIM / 2.0, constants.YDIM * 2.0 / 3.0, 0])
    mod.set_weight(constants.FITNESS_APPROACH_FLAG, 1)
    mod.deploy('lamarck', 'neatq')
    OpenNero.enable_ai()


def ModTick(dt):
    global mod
    global step
    step += 1
    if step == 100000:
        print "ENVIRONMENTAL CHANGE AT ", step
        mod.change_flag([constants.XDIM / 2.0, constants.YDIM / 3.0, 0])
        mod.set_weight(constants.FITNESS_APPROACH_FLAG, -1)
        mod.save_team(os.path.expanduser('~/.opennero/adapt_mid.json'))
    if step > 200000:
        mod.save_team(os.path.expanduser('~/.opennero/adapt_final.json'))
        OpenNero.getSimContext().killGame()
    if step % 100 == 0:
        LogStats()


def LogStats():
    agents = mod.environment.states.keys()
    fitnesses = [agent.org.fitness for agent in agents]
    min_fit = min(fitnesses)
    max_fit = max(fitnesses)
    mean_fit = sum(fitnesses) / len(fitnesses)
    print "Minimum Fitness: %.2f Maximum Fitness: %.2f Average Fitness: %.2f" %\
        (min_fit, max_fit, mean_fit)
