import sys
import json
import string
import io
import operator
import random
import constants
import OpenNero


def factory_class(ai):
    return ai_map.get(ai, NEATAgent)


def factory(ai, *args):
    return factory_class(ai)(*args)


class NeroAgent(object):
    """
    base class for nero agents
    """
    group = 'Agent'

    def __init__(self, team_type, *args):
        self.team_type = team_type

    def initialize(self, init_info):
        self.actions = init_info.actions
        self.sensors = init_info.sensors
        self.rewards = init_info.reward
        return True

    def start(self, time, sensors):
        pass

    def end(self, sensors, reward):
        return True

    def destroy(self):
        return True

    def ai_label(self):
        return inv_ai_map.get(self.__class__, 'none')

    def args(self):
        return []


class NEATAgent(NeroAgent, OpenNero.AgentBrain):
    num_inputs = constants.N_SENSORS + 1
    num_outputs = constants.N_ACTIONS
    base_genome = OpenNero.Genome(num_inputs, num_outputs, 0, 0)
    count = 0

    def __init__(self, team_type, *args):
        """
        Create an agent brain
        """
        # this line is crucial, otherwise the class is not recognized as an
        # AgentBrainPtr by C++
        OpenNero.AgentBrain.__init__(self)

        NeroAgent.__init__(self, team_type)

        self.omit_friend_sensors = False

        if len(args) > 0:
            stream = io.BytesIO(string.join(args, '\n').encode('utf-8'))
            self.org = OpenNero.Organism(stream)
        else:
            NEATAgent.count += 1
            genome = self.base_genome.clone(NEATAgent.count, 1)
            self.org = OpenNero.Organism(0, genome, 1)

    def args(self):
        return str(self.org).split('\n')

    def start(self, time, sensors):
        """
        start of an episode
        """
        self.org.time_alive += self.step
        return self.network_action(sensors)

    def act(self, time, sensors, reward):
        """
        a state transition
        """
        # return action
        self.org.time_alive = self.step
        self.org.fitness = self.fitness[0] / self.step
        self.last_reward = reward[0]
        action = self.network_action(sensors)
        return self.interpret_action(action)

    def set_display_hint(self):
        """
        set the display hint above the agent's head (toggled with F2)
        """
        display_hint = constants.getDisplayHint()
        if display_hint:
            if display_hint == 'fitness':
                self.state.label = '%.2f' % self.org.fitness
            elif display_hint == 'reward':
                self.state.label = '%.2f' % self.last_reward
            elif display_hint == 'time alive':
                self.state.label = str(self.org.time_alive)
            elif display_hint == 'hit points':
                self.state.label = ''.join('.' for i in range(
                    int(5 * OpenNero.get_environment().get_hitpoints(self))))
            elif display_hint == 'id':
                self.state.label = str(self.org.genome.id)
            elif display_hint == 'champion':
                if self.org.champion:
                    self.state.label = 'champ!'
                else:
                    self.state.label = ''
            elif display_hint == 'rank':
                self.state.label = str(self.org.rank)
            elif display_hint == 'debug':
                self.state.label = str(
                    OpenNero.get_environment().get_state(self))
            else:
                self.state.label = '?'
        else:
            # the first time we switch away from displaying stuff,
            # change the window caption
            if self.state.label:
                self.state.label = ""

    def load_sensors(self, sensors):
        assert(len(sensors) == constants.N_SENSORS)

        if self.omit_friend_sensors:
            for idx in constants.SENSOR_INDEX_FRIEND_RADAR:
                sensors[idx] = 0

        self.org.net.load_sensors(
            list(self.sensors.normalize(sensors)) + [constants.NEAT_BIAS])

    def interpret_action(self, actions):
        dn_actions = self.actions.denormalize(actions)
        if dn_actions[constants.ACTION_INDEX_ZERO_FRIEND_SENSORS] > 0.5:
            self.omit_friend_sensors = True
        else:
            self.omit_friend_sensors = False
        return dn_actions

    def network_action(self, sensors):
        """
        Take the current network
        Feed the sensors into it
        Activate the network to produce the output
        Collect and interpret the outputs as valid actions
        """
        self.load_sensors(sensors)
        self.org.net.activate()

        actions = self.actions.get_instance()
        for i in range(len(actions)):
            actions[i] = self.org.net.outputs[i].active_out
        return actions


class NEATQAgent(NEATAgent):
    num_outputs = NEATAgent.num_outputs + 1
    sigma = 0.05
    alpha = 0.95

    def __init__(self, *args):
        NEATAgent.__init__(self, *args)
        self.last_action = None
        self.last_sensors = None

    def act(self, time, sensors, reward):
        action = NEATAgent.act(self, time, sensors, reward)

        if self.last_action:
            q_new = self.org.net.outputs[-1].active_out
            q_old = self.org.net.outputs[-1].active_out_td
            q_err = (1 - self.alpha) * (reward[0] + self.alpha * q_new) - q_old

            last_network = [
                node.active_out_td for node in self.org.net.outputs]
            errors = [new - old for new,
                      old in zip(last_network, self.last_action)]
            if q_err < 0:
                errors = [e * -1.0 for e in errors]

            errors += [q_err]
            self.org.net.load_sensors(list(self.sensors.normalize(
                self.last_sensors)) + [constants.NEAT_BIAS])
            self.org.net.load_errors(errors)
            self.org.net.backprop()

        self.last_sensors = sensors
        self.last_action = self.actions.normalize(action)
        return action

    def network_action(self, sensors):
        actions = NEATAgent.network_action(self, sensors)
        for i in range(len(actions)):
            actions[i] += random.gauss(0, self.sigma)
            actions[i] = max(actions[i], 0)
            actions[i] = min(actions[i], 1)
        return actions


class QLearningAgent(NeroAgent, OpenNero.QLearningBrain):
    """
    QLearning agent.
    """

    def __init__(self, team_type, gamma=0.8, alpha=0.8, epsilon=0.1,
                 action_bins=3, state_bins=5,
                 num_tiles=0, num_weights=0):
        OpenNero.QLearningBrain.__init__(
            self, gamma, alpha, epsilon,
            action_bins, state_bins,
            num_tiles, num_weights)
        NeroAgent.__init__(self, team_type)

    def set_display_hint(self):
        """
        set the display hint above the agent's head (toggled with F2)
        """
        display_hint = constants.getDisplayHint()
        if display_hint:
            if display_hint == 'fitness':
                self.state.label = '%.2g' % self.fitness[0]
            elif display_hint == 'time alive':
                self.state.label = str(self.step)
            elif display_hint == 'hit points':
                self.state.label = ''.join('.' for i in range(
                    int(5 * OpenNero.get_environment().get_hitpoints(self))))
            elif display_hint == 'id':
                self.state.label = str(self.state.id)
            elif display_hint == 'debug':
                self.state.label = str(
                    OpenNero.get_environment().get_state(self))
            else:
                self.state.label = '?'
        else:
            # the first time we switch away from displaying stuff,
            # change the window caption
            if self.state.label:
                self.state.label = ""


class Turret(NeroAgent, OpenNero.AgentBrain):
    """
    Simple Rotating Turret
    """
    group = 'Turret'

    def __init__(self, team_type, *args):
        OpenNero.AgentBrain.__init__(self)
        NeroAgent.__init__(self, team_type, *args)

    def start(self, time, sensors):
        self.state.label = 'Turret'
        a = self.actions.get_instance()
        a[0] = a[1] = a[2] = a[3] = 0
        return a

    def act(self, time, sensors, reward):
        a = self.actions.get_instance()
        a[0] = 0
        a[1] = 0.1
        a[2] = 1
        a[3] = 0
        return a

ai_map = {
    'neat': NEATAgent,
    'qlearning': QLearningAgent,
    'neatq': NEATQAgent,
    'turret': Turret
}

inv_ai_map = {v: k for k, v in ai_map.items()}
