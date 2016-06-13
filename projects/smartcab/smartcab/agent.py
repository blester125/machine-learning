import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import sys

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, a=0.8):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # Initialize any additional variables here
        self.QTable = Table(a)
        self.total_reward = 0
        self.state = ""
        self.trial = 0
        self.time = 0
        self.movement = 0
        #self.out = open("output.txt", "w")

    def argmax_print(self):
        for state in self.QTable.table.keys():
            print state + ": " + str(argmax(self.QTable.table[state]))

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # Prepare for a new trip; reset any variables here, if required
        print "Total reward from the last trip:"
        print self.total_reward
        #self.out.write("\nTrial Number: " + str(self.trial) +"\n")
        print "Total time for this trip:"
        print self.time
        print "Total movement in this trip:"
        print self.movement, '\n'
        self.total_reward = 0
        self.trial += 1
        self.time = 0
        self.movement = 0

    def update(self, t):
        print "\n********************Start Turn********************"
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        # organize input
        light = inputs['light']
        oncoming = inputs['oncoming']
        right = inputs['right']
        left = inputs['left']
        self.state = inputs_to_state(light, left, oncoming, self.next_waypoint)
        print "Inputs:"
        print "light - " + str(light)
        print "left - " + str(left)
        print "oncoming - " + str(oncoming)
        print "right - " + str(right)
        print "deadline - " + str(deadline)
        print "waypoint - " + str(self.next_waypoint)
        print "State - " + str(self.state)
        print ""

        # Select action according to your policy
        action = self.QTable.select_action(self.state, self.next_waypoint)
        if action != None:
            self.movement += 1
        self.time += 1
        # Random code from the inital question
        #moves = [None, 'forward', 'right', 'left']
        #action = random.choice(moves)
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward < 0:
            #self.out.write("Negative reward gotten\n")
            print "Negative reward gotten"
        self.total_reward += reward
        # Learn policy based on state, action, reward
        self.QTable.update(self.state, action, reward)

        # print "LearningAgent.update(): \nwaypoint = {}, \ndeadline = {}, \ninputs = {}, \naction = {}, \nreward = {}".format(self.next_waypoint, deadline, inputs, action, reward)  # [debug]

class Table():
    def __init__(self, alpha=1.0, gamma=0.5):
        self.alpha = alpha
        self.gamma = gamma
        self.table = self.initialize_table()

    def Print(self):
        print self.table

    def initialize_table(self):
        table = {}
        for light in ['green', 'red']:
            for left in ['forward', 'left', 'right', None]:
                for oncoming in ['forward', 'left', 'right', None]:
                    for next_waypoint in ['forward', 'left', 'right', None]:
                        state = inputs_to_state(light,left,oncoming,next_waypoint)
                        table[state] = {}
                        for action in [None, 'forward', 'left', 'right']:
                            # The initial values are set to 4 to help avoid the agent 
                            # getting stuck in local minimum
                            # This is called Optimistic Initialization 
                            table[state][action] = 4.0
        return table

    def get_alpha(self):
        return self.alpha

    def get_gamma(self):
        return self.gamma

    def get_value(self, state=None, action=None):
        return self.table[state][action]

    def set_value(self, state=None, action=None, value=0.0):
        self.table[state][action] = value

    def update(self, state=None, action=None, reward=0.0):
        old_value = self.get_value(state, action)
        # This equation is used for reward discounting
        #new_value = old_value * (1 - self.get_alpha()) + self.get_alpha() * (reward + self.get_gamma() * old_value)
        new_value = old_value * (1 - self.get_alpha()) + (self.get_alpha() * reward)
        self.set_value(state, action, new_value)

    def select_action(self, state=None, next=None):
        actions = [None, 'forward', 'left', 'right']
        values = {}
        for action in actions:
            values[action] = self.get_value(state, action)
        maxs = []
        max_value = values[random.choice(values.keys())]
        print "Values in the Q table for each action:"
        for value in values:
            print str(value) + ": " + str(values[value])
            if values[value] >= max_value:
                max_value = values[value]
        for value in values:
            if values[value] == max_value:
                maxs.append(value)
        print "List of actions with the same values"
        print maxs
        if len(maxs) > 1:
            print "---------------Random Used--------------"
        best_action = random.choice(maxs)
        print "Action choosen: " + str(best_action)
        return best_action

def inputs_to_state(light=None,
                   left=None,
                   oncoming=None,
                   next_waypoint=None):
    return "{}|{}|{}|{}".format(light,left,oncoming,next_waypoint)

def argmax(dic):
    action = ""
    max_value = dic[random.choice(dic.keys())]
    for value in dic:
        print str(value) + ": " + str(dic[value])
        if dic[value] >= max_value:
            max_value = dic[value]
            action = value
    return action

def run():
    """Run the agent for a finite number of trials."""

    # Code for testing multiple alphas
    """
    i = 0
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
    for alpha in alphas:
        print "********************Run " + str(i) + " Alpha is " + str(alpha) +"********************"
        # Set up environment and agent
        e = Environment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent, alpha)  # create agent
        e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
        # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

        # Now simulate it
        sim = Simulator(e, update_delay=0.0001, display=False)  # create simulator (uses pygame when display=True, if available)
        # NOTE: To speed up simulation, reduce update_delay and/or set display=False

        sim.run(n_trials=100)  # run for a specified number of trials
        # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
        i += 1
    """
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    #a.argmax_print()
    

if __name__ == '__main__':
    run()
