###########################################################################

# Benjamin Wedeen, bewe0593@colorado.edu
# Professor Orit Peleg, orit.peleg@colorado.edu

# Code for CSCI 4314/5314 Dynamic Models in Biology Final Project

# Agent-Based COVID-19 Simulator - Base code from Nicolas Bohorquez using Mesa framework

###########################################################################
import matplotlib
import mesa
import pandas as pd
import enum
import math
import numpy as np
import networkx as nx
import random as random
import matplotlib as mpl
from celluloid import Camera
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.batchrunner import BatchRunner
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

#---getEuclideanDistance---#
#Gets distance between points x and y
def getEuclideanDistance(x, y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x,y)]))

#---InfectionStatus---#
#Assign a corresponding number to each infectionStatus based on the SIR model
class InfectionStatus(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = -1
    RECOVERED = 0

#---EconomicStatus---#
#Assign a corresponding number to each economicStatus
class EconomicStatus(enum.IntEnum):
    LOWER_CLASS = 2
    MIDDLE_CLASS = 1
    UPPER_CLASS = 0
    NO_CLASS = -1

#---myAgent---#
#An agent in the simulation
class myAgent(Agent):
    def __init__(self, unique_id, model, xInit, yInit, destinations, economicStatus = []):
        super().__init__(unique_id, model)
        self.init_pos=(xInit, yInit)
        self.target_pos = None
        self.destinations = destinations
        self.infection_time = 0
        self.infected_at = 0

        #Initialize agents as Suceptible
        self.infectionStatus = InfectionStatus.SUSCEPTIBLE
        #Initialize agent economic status, influences infectionStatus and death_probability
        self.EconomicStatus = economicStatus

    def step(self):
        self.check()
        self.interact()
        self.move()

    #Check to see if infected agent dies from infection
    def check(self):
        if self.infectionStatus == InfectionStatus.INFECTED:
            death_probability = self.model.death_probability
            if(self.EconomicStatus == EconomicStatus.NO_CLASS):
                self.model.death_probability += 0
            if(self.EconomicStatus == EconomicStatus.UPPER_CLASS):
                self.model.death_probability += 0
            if (self.EconomicStatus == EconomicStatus.MIDDLE_CLASS):
                self.model.death_probability += 0.00
            if (self.EconomicStatus == EconomicStatus.LOWER_CLASS):
                self.model.death_probability += 0.00
            np.random.seed = self.random.seed
            #Roll to check if selected agent dies this step
            is_alive = np.random.choice([0,1], p=[death_probability, 1-death_probability])
            #If selected agent dies, remove them from the simulation and increase death count by 1
            if is_alive == 0:
                self.model.schedule.remove(self)
                self.model.deaths += 1
            #If selected agent does not die, set InfectionStatus to recovered
            elif self. model.schedule.time - self.infected_at >= self.model.treatment_period:
                self.infectionStatus = InfectionStatus.RECOVERED

    #Rules for selecting next movement for selected agent
    def move(self):
        self.set_target_pos()
        #Get possible target agent movements in relation to target agent surrondings (no two agents can occupy the same space on the grid)
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore = True, include_center = False)
        new_position = self.select_new_pos(possible_steps)
        self.model.grid.move_agent(self, new_position)

    #Rules for selected agent interaction between surronding agents
    def interact(self):
        #Get info on agents surronding target agent
        contacts = self.model.grid.get_cell_list_contents([self.pos])
        for c in contacts:
            #If selected agent is infected and around other agents, roll to see if the selected agent infects each surronding agent
            if self.infectionStatus is InfectionStatus.INFECTED and c.infectionStatus is not InfectionStatus.INFECTED:
                if(self.EconomicStatus == EconomicStatus.NO_CLASS):
                    self.model.infection_probability += 0
                if (self.EconomicStatus == EconomicStatus.UPPER_CLASS):
                    self.model.infection_probability -= 0.05
                if (self.EconomicStatus == EconomicStatus.MIDDLE_CLASS):
                    self.model.infection_probability += 0.10
                if (self.EconomicStatus == EconomicStatus.LOWER_CLASS):
                    self.model.infection_probability *= 1.7
                infect = self.random.random() <= self.model.infection_probability
                #If selected agent is infected and infects another agent, set other agent's status to infected, and record the time of infection and spreader
                if infect == True:
                    c.infectionStatus = InfectionStatus.INFECTED
                    c.infected_at = self.model.schedule.time
                    self.model.addInfection(self.unique_id, c.unique_id)

    def set_target_pos (self):
        #If agent is at home, have them go to a destination
        if self.pos == self.init_pos:
            self.target_pos = self.random.choice(self.destinations)
        #If target is at a destination, have them go home
        elif self.pos == self.target_pos:
            self.target_pos = self.init_pos

    def select_new_pos(self, possible_steps):
        if self.infectionStatus == InfectionStatus.INFECTED:
            #Calculate whether selected agent is showing symptoms of virus
            has_symptoms = self.model.schedule.time - self.infection_time >= self.model.incubation_period
            # Infected agents with symptoms who are home will stay at home
            if self.pos == self.init_pos and has_symptoms == True:
                return self.init_pos
            # Infected agents  with symptoms will go home and quarantine if they are not home already
            elif self.pos != self.init_pos and has_symptoms == True:
                self.target_pos = self.init_pos

        return self.calculate_new_pos(possible_steps)

    def calculate_new_pos(self, possible_steps):
        next_step = possible_steps[0]
        next_dist = getEuclideanDistance(self.target_pos, next_step)
        for step in possible_steps:
            dist = getEuclideanDistance(self.target_pos, step)
            if dist < next_dist:
                next_step = step
        return next_step

class CovidModel(Model):
    def __init__(self, N, width, height, infection_probability, death_base_probability, incubation_period, treatment_period,
                 sim_destinations, economicStatus, seed=None):
        self.agentCount = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.infection_probability = infection_probability
        self.death_probability = death_base_probability
        self.incubation_period = incubation_period
        self.treatment_period = treatment_period
        self.deaths = 0
        self.DG = nx.DiGraph()
        self.running = True
        self.economicStatus = economicStatus
        # adds a fixed number of possible destinations
        self.destinations = []
        for idx in range(sim_destinations):
            self.destinations.append((self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))

        # Create agents
        for i in range(self.agentCount):
            #Create random coordinates to set as agent initial position
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            #Initialize a new agent
            a = myAgent(i, self, x, y, self.destinations, self.economicStatus)
            #Set the last created agent to be patient zero
            if i == self.agentCount - 1:
                a.infectionStatus = InfectionStatus.INFECTED
                self.DG.add_nodes_from([(i, {"color": "green"})])
            else:
                self.DG.add_nodes_from([(i, {"color": "orange"})])
            self.schedule.add(a)
            # Add the agent to a random grid cell
            self.grid.place_agent(a, (x, y))

        def compute_S(model):
            agents = len([agent.infectionStatus for agent in model.schedule.agents if agent.infectionStatus == InfectionStatus.SUSCEPTIBLE])
            return agents

        def compute_I(model):
            agents = len([agent.infectionStatus for agent in model.schedule.agents if agent.infectionStatus == InfectionStatus.INFECTED])
            return agents

        def compute_R(model):
            agents = len([agent.infectionStatus for agent in model.schedule.agents if agent.infectionStatus == InfectionStatus.RECOVERED])
            return agents

        def compute_D(model):
            agents = model.agentCount - len(model.schedule.agents)
            return agents

        def compute_degree(model):
            degree = np.median([t[1] for t in model.DG.degree()])
            return degree

        self.datacollector = DataCollector(
            model_reporters={"S": compute_S, "I": compute_I, "R": compute_R, "D": compute_D, "AvgDegree": compute_degree}
            , agent_reporters={"InfectionStatus": "infectionStatus", "Position": "pos"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def addInfection(self, agentIdSrc, agentIdDes):
        self.DG.add_edge(agentIdSrc, agentIdDes)


maxX = maxY = 20
sim_steps = 100
sim_agents = 200
sim_infection_prob = 0.2
sim_death_base_probability = 0.05
sim_incubation_period = 3
sim_treatment_period = 14
sim_destinations = 10
sim_rand_seed = 1
sim_economic_status = EconomicStatus.LOWER_CLASS

model = CovidModel(sim_agents, maxX, maxY, sim_infection_prob, sim_death_base_probability, sim_incubation_period, sim_treatment_period, sim_destinations, sim_economic_status,seed = sim_rand_seed)


for i in range(sim_steps):
    model.step()
sir_df = model.datacollector.get_model_vars_dataframe()

agents_data = model.datacollector.get_agent_vars_dataframe()
agents_data[['xPos', 'yPos']] = pd.DataFrame(agents_data["Position"].to_list(), index=agents_data.index, columns=['xPos', 'yPos'])
df_agents = (agents_data.reset_index(level=0)).reset_index(level=0)

fig, ax = plt.subplots(1, figsize=(8, 6))
ax.set_xlim(( 0, maxX-1))
ax.set_ylim(( 0, maxY-1))
for st in model.destinations:
    ax.scatter(st[0],st[1],s=300,color='black', marker=r'$\diamond$', label="Destination")
curr_step = df_agents.loc[(df_agents['Step']==0) & (df_agents['InfectionStatus']==InfectionStatus.SUSCEPTIBLE)]
ax.scatter(curr_step['xPos'],curr_step['yPos'],alpha=0.5,color='green')
curr_step = df_agents.loc[(df_agents['Step']==0) & (df_agents['InfectionStatus']==InfectionStatus.INFECTED)]
ax.scatter(curr_step['xPos'],curr_step['yPos'],alpha=0.5,color='red')
plt.grid(visible=True, which='major', color='#666666', linestyle='-')


fig, ax = plt.subplots(1, figsize=(8, 6))
ax.set_xlabel = 'Step'
ax.set_title = 'SIR behaviour'
ax.plot(range(sim_steps), sir_df['AvgDegree'], label='Average node degree', color='red')
ax.legend()

fig, ax = plt.subplots(1, figsize=(8, 6))
ax.set_xlabel = 'Step'
ax.set_title = 'SIR behaviour'
ax.plot(range(sim_steps), sir_df['S'], label='S', color='green')
ax.plot(range(sim_steps), sir_df['I'], label='I', color='orange')
ax.plot(range(sim_steps), sir_df['R'], label='R', color='blue')
ax.plot(range(sim_steps), sir_df['D'], label='Deaths', color='black')
ax.legend()



fig, ax = plt.subplots(1, figsize=(8, 6))
camera = Camera(fig)
for s in range(sim_steps):
    for st in model.destinations:
        ax.scatter(st[0],st[1],s=300,color='black', marker=r'$\diamond$', label="Destination")
    #plot susceptibles
    curr_step = df_agents.loc[(df_agents['Step']==s) & (df_agents['InfectionStatus']==InfectionStatus.SUSCEPTIBLE)]
    ax.scatter(curr_step['xPos'],curr_step['yPos'],alpha=0.5,color='green')
    curr_step = df_agents.loc[(df_agents['Step']==s) & (df_agents['InfectionStatus']==InfectionStatus.INFECTED)]
    ax.scatter(curr_step['xPos'],curr_step['yPos'],alpha=0.5,color='red')
    curr_step = df_agents.loc[(df_agents['Step']==s) & (df_agents['InfectionStatus']==InfectionStatus.RECOVERED)]
    ax.scatter(curr_step['xPos'],curr_step['yPos'],alpha=0.5,color='cyan')
    camera.snap()
anim = camera.animate(blit=True)


plt.show()

