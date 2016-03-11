# import all client and server scripts
import NERO.module as module
import NERO.client
import NERO.constants as constants
import common
import common.menu_utils
import OpenNero
import os

mod = None
tick = None
step = 0


def ModMain(mode=""):
    NERO.client.ClientMain()


def ModTick(dt):
    global tick
    if tick:
        tick(dt)


def RegisterTick(tick_func):
    global tick
    tick = tick_func


def LogStats():
    global mod
    agents = mod.environment.states.keys()
    fitnesses = [agent.org.fitness for agent in agents]
    min_fit = min(fitnesses)
    max_fit = max(fitnesses)
    mean_fit = sum(fitnesses) / len(fitnesses)
    print "Minimum Fitness: %.2f Maximum Fitness: %.2f Average Fitness: %.2f" %\
        (min_fit, max_fit, mean_fit)


def adapt_tick(dt):
    global mod
    global step
    epoch = 50000
    step += 1
    if step == epoch:
        mod.change_flag([constants.XDIM / 2.0, constants.YDIM / 3.0, 0])
        mod.set_weight(constants.FITNESS_APPROACH_FLAG, -1)
        mod.save_team(os.path.expanduser('~/.opennero/adapt_mid.json'))
    if step > (epoch * 2):
        mod.save_team(os.path.expanduser('~/.opennero/adapt_final.json'))
        OpenNero.getSimContext().killGame()
    if step % 100 == 0:
        LogStats()


def Adapt(team_ai, agent_ai):
    global mod
    if not mod:
        mod = NERO.module.getMod()

    OpenNero.disable_ai()
    RegisterTick(adapt_tick)

    mod.change_flag([constants.XDIM / 2.0, constants.YDIM * 2.0 / 3.0, 0])
    mod.set_weight(constants.FITNESS_APPROACH_FLAG, 1)
    mod.deploy(team_ai, agent_ai)

    OpenNero.enable_ai()
