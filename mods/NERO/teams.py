import json

import constants
import OpenNero
import agent as agents


def factory(ai, *args):
    cls = ai_map.get(ai, NeroTeam)
    return cls(*args)


class TeamEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, NeroTeam):
            return {
                'team_ai': inv_ai_map.get(obj.__class__, 'none'),
                'agents': [
                    {
                        'agent_ai': agent.ai_label(),
                        'args': agent.args()
                    }
                    for agent in obj.agents
                ]
            }
        return json.JSONEncoder.default(self, obj)


def as_team(team_type, dct):
    if 'team_ai' in dct:
        team = factory(dct['team_ai'], team_type)
        for a in dct['agents'][:constants.pop_size]:
            team.create_agent(a['agent_ai'], *a['args'])
        return team
    return dct


class NeroTeam(object):
    """
    Basic NERO Team
    """

    def __init__(self, team_type):
        self.team_type = team_type
        self.color = constants.TEAM_LABELS[team_type]
        self.agents = set()
        self.dead_agents = set()

    def create_agents(self, ai):
        for _ in range(constants.pop_size):
            self.create_agent(ai)

    def create_agent(self, ai, *args):
        a = agents.factory(ai, self.team_type, *args)
        self.add_agent(a)
        return a

    def add_agent(self, a):
        self.agents.add(a)

    def kill_agent(self, a):
        self.agents.remove(a)
        self.dead_agents.add(a)

    def is_episode_over(self, agent):
        return False

    def reset(self, agent):
        pass

    def start_training(self):
        pass

    def stop_training(self):
        pass

    def is_destroyed(self):
        return len(self.agents) == 0 and len(self.dead_agents) > 0

    def reset_all(self):
        self.agents |= self.dead_agents
        self.dead_agents = set()


class RTNEATTeam(NeroTeam):

    def __init__(self, team_type):
        NeroTeam.__init__(self, team_type)
        self.pop = OpenNero.Population()
        self.rtneat = OpenNero.RTNEAT("data/ai/neat-params.dat",
                                      self.pop,
                                      constants.DEFAULT_LIFETIME_MIN,
                                      constants.DEFAULT_EVOLVE_RATE)
        self.generation = 1

    def add_agent(self, a):
        NeroTeam.add_agent(self, a)
        self.pop.add_organism(a.org)

    def start_training(self):
        OpenNero.set_ai('rtneat-%s' % self.team_type, self.rtneat)

    def stop_training(self):
        OpenNero.set_ai('rtneat-%s' % self.team_type, None)

    def is_episode_over(self, agent):
        return agent.org.eliminate

    def reset(self, agent):
        if agent.org.eliminate:
            agent.org = self.rtneat.reproduce_one()
        else:
            agent.org.update_phenotype()

    def reset_all(self):
        NeroTeam.reset_all(self)
        # TODO: Epoch can segfault without fitness differentials
        if any([agent.org.fitness > 0 for agent in self.agents]):
            self.generation += 1
            self.pop.epoch(self.generation)
            for agent, org in zip(self.agents, self.pop.organisms):
                agent.org = org


class LamarckianRTNEATTeam(RTNEATTeam):
    def reset(self, agent):
        if agent.org.eliminate:
            agent.org = self.rtneat.reproduce_lamarckian()
        else:
            agent.org.update_phenotype()


ai_map = {
    'rtneat': RTNEATTeam,
    'lamarck': LamarckianRTNEATTeam,
    'none': NeroTeam
}

inv_ai_map = {v: k for k, v in ai_map.items()}
