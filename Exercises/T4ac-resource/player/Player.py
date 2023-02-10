#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from time import sleep
import random
import numpy as np
from pathlib import Path
import gurobipy as gp
from gurobipy import *

import pickle

import GridMap as gmap

PURSUER = 1
EVADER = 2

GREEDY = "GREEDY"
MONTE_CARLO = "MONTE_CARLO"
VALUE_ITERATION = "VALUE_ITERATION"

def compute_nash(matrix, only_value=False, minimize=False):
    """
    Method to calculate the value-iteration policy action
    https://cw.fel.cvut.cz/b192/_media/courses/ko/01_gurobi.pdf

    Parameters
    ----------
    matrix: n times m array of floats
        Game utility matrix


    Returns
    -------
    value:float
        computed value of the game
    strategy:float[n]
        probability of player 1 playing each action in nash equilibrium

    Purpose of the function is to compute nash equilibrium of any matrix game.
    The function should compute strategy for the row player assuming he is the maximizing player.
    If you want to use it to compute the strategy of the other player all you have to do is negate and transpose the matrix.
    """
    # START OF MY CODE
    # I have to use the Gurobi library to solve the linear program
    # I have to create a model
    num_actions = matrix.shape[0]

    # Create a new Gurobi model
    model = gp.Model()
    model.setParam("OutputFlag", 0) #Mute the message

    # Set the objective: maximize / minimize v
    v = model.addVar(vtype=gp.GRB.CONTINUOUS, name="v", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY) # up=upper bound, lb=lower bound
    model.setObjective(v, gp.GRB.MAXIMIZE)

    # Add variables for the strategies of the two players
    x = model.addVars(num_actions,lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")

    # Add constraints
    model.addConstrs((gp.quicksum(x[i] * matrix[i, j] for i in range(num_actions))>=v for j in range(matrix.shape[1])))
    model.addConstr(gp.quicksum(x[i] for i in range(num_actions)) == 1)
    model.addConstrs((x[i] >= 0 for i in range(num_actions)))

    # Solve the model
    model.optimize()

    # Check if an optimal solution was found
    if model.status != gp.GRB.OPTIMAL:
        raise Exception("Gurobi optimization failed with status " + str(model.status))

    # Return the value and strategies
    return model.objVal, [x[i].x for i in range(num_actions)]
    
    # END OF MY CODE
    #maximizer_actions = matrix.shape[0]
    #return 0, [1./maximizer_actions]*maximizer_actions
    

class Player:
    def __init__(self, robots, role, policy=GREEDY, color='r', epsilon=1, 
                 timeout=5.0, game_name=None):
        """ constructor of the Player class
        Args: robots: list((in,int)) - coordinates of individual player's robots
              role: PURSUER/EVADER - player's role in the game
              policy: GREEDY/MONTE_CARLO/VALUE_ITERATION - player's policy, 
              color: string - player color for visualization
              epsilon: float - [0,1] epsilon value for greedy policy
              timeout: float - timout for MCTS policy
              game_name: string - name of the currently played game 
        """
        # list of the player's robots
        self.robots = robots[:]
        # next position of the player's robots
        self.next_robots = robots[:]

        if role == "EVADER":
            self.role = EVADER
        elif role == "PURSUER":
            self.role = PURSUER
        else:
            raise ValueError('Unknown player role')

        # selection of the policy
        if policy == GREEDY:
            self.policy = self.greedy_policy
        elif policy == MONTE_CARLO:
            self.policy = self.monte_carlo_policy
            self.timeout = timeout * len(self.robots) # MCTS planning timeout
            self.tree = {}
            self.max_depth = 10
            self.step = 0
            self.max_steps = 100
            self.beta = 0.95
            self.c = 1
        elif policy == VALUE_ITERATION:
            self.policy = self.value_iteration_policy
            # values for the value iteration policy
            self.loaded_policy = None
            self.gamma = 0.95
        else:
            raise ValueError('Unknown policy')

        #parameters
        self.color = color # color for plotting purposes
        self.game_name = game_name # game name for loading vi policies


    #####################################################
    # Game interface functions
    #####################################################
    def add_robot(self, pos):
        """ method to add a robot to the player
        Args: pos: (int,int) - position of the robot
        """
        self.robots.append(pos)
        self.next_robots.append(pos)

    def del_robot(self, pos):
        """ method to remove the player's robot 
        Args: pos: (int,int) - position of the robot to be removed
        """
        idx = self.robots.index(pos)
        self.robots.pop(idx)
        self.next_robots.pop(idx)

    def calculate_step(self, gridmap, evaders, pursuers):
        """ method to calculate the player's next step using selected policy
        Args: gridmap: GridMap - map of the environment
              evaders: list((int,int)) - list of coordinates of evaders in the 
                            game (except the player's robots, if he is evader)
              pursuers: list((int,int)) - list of coordinates of pursuers in 
                       the game (except the player's robots, if he is pursuer)
        """        
        self.policy(gridmap, evaders, pursuers)

    def take_step(self):
        """ method to perform the step 
        """
        self.robots = self.next_robots[:]

    #####################################################
    # GREEDY POLICY
    #####################################################
    def greedy_policy(self, gridmap, evaders, pursuers, epsilon=1):
        """ Method to calculate the greedy policy action
        Args: gridmap: GridMap - map of the environment
              evaders: list((int,int)) - list of coordinates of evaders in the 
                            game (except the player's robots, if he is evader)
              pursuers: list((int,int)) - list of coordinates of pursuers in 
                       the game (except the player's robots, if he is pursuer)
              epsilon: float (optional) - optional epsilon-greedy parameter
        """
        self.next_robots = self.robots[:]

        # START OF MY CODE
        
        # for each of player's robots plan their actions
        for idx in range(0, len(self.robots)):
            robot = self.robots[idx]
            neighbors = gridmap.neighbors4(robot) # extract possible coordinates to go (actions)

            ##################################################
            # RANDOM Policy
            ##################################################
            if random.random()>epsilon:
                random.shuffle(neighbors) # randomness in neighbor selection
                pos_selected = neighbors[0] # select random goal

                # select the next action based on own role
                if self.role == PURSUER:
                    self.next_robots[idx] = pos_selected

                if self.role == EVADER:
                    self.next_robots[idx] = pos_selected

            ##################################################
            # NORMAL Policy
            ##################################################       
            else:
                if self.role == PURSUER:
                    # shortest distance
                    min_val = float('inf')
                    for pos in neighbors:
                        val = -float('inf')
                        for evader in evaders:
                            val = max(val,gridmap.dist(pos, evader))
                        if val < min_val:
                            min_val = val
                            pos_selected = pos
                    # extract possible coordinates to go (actions)
                    self.next_robots[idx] = pos_selected


                if self.role == EVADER:
                    # longest distance
                    max_val = -float('inf')
                    for pos in neighbors:
                        val = float('inf')
                        for pursuer in pursuers:
                            val = min(val,gridmap.dist(pos, pursuer))
                        if val > max_val:
                            max_val = val
                            pos_selected = pos
                    self.next_robots[idx] = pos_selected

        # END OF MY CODE
                

    #####################################################
    # VALUE ITERATION POLICY
    #####################################################
    def init_values(self, gridmap):
        mapping_i2c = {}
        mapping_c2i = {}
        count = 0
        for i in range(gridmap.width):
            for j in range(gridmap.height):
                if gridmap.passable((i, j)):
                    mapping_i2c[count] = (i, j)
                    mapping_c2i[(i, j)] = count
                    count += 1
        return mapping_i2c, mapping_c2i, count

    def random_policy(self, coord_state, gridmap, mapping_c2i, role):
        a, b, c = coord_state
        neigh_a = gridmap.neighbors4(a)
        neigh_b = gridmap.neighbors4(b)
        neigh_c = gridmap.neighbors4(c)        
        if role == PURSUER:
            combined_actions = []
            for action_one in neigh_b:
                for action_two in neigh_c:
                    combined_actions.append((mapping_c2i[action_one], mapping_c2i[action_two]))
            return(combined_actions, [1/len(combined_actions)]*len(combined_actions))
        else:
            combined_actions = []
            for action in neigh_a:
                combined_actions.append(mapping_c2i[action])
            return(combined_actions, [1/len(combined_actions)]*len(combined_actions))

    def compute_random_policy(self, gridmap):
        mapping_i2c, mapping_c2i, N = self.init_values(gridmap)
        values = np.zeros((N,N,N))
        policy_e = {}
        policy_p = {}
        for a in range(N):
            for b in range(N):
                for c in range(N):
                    coord_state = (mapping_i2c[a], mapping_i2c[b], mapping_i2c[c])
                    policy_e[(a,b,c)] = self.random_policy(coord_state, gridmap, mapping_c2i, EVADER)
                    policy_p[(a,b,c)] = self.random_policy(coord_state, gridmap, mapping_c2i, PURSUER)
        return values, policy_e, policy_p, mapping_i2c, mapping_c2i

    def value_iteration_policy(self, gridmap, evaders, pursuers):
        """
        Method to calculate the value-iteration policy action
        https://cw.fel.cvut.cz/b221/courses/uir/hw/t4c-vi

        Parameters
        ----------
        gridmap: GridMap
            Map of the environment
        evaders: list((int,int))
            list of coordinates of evaders in the game (except the player's robots, if he is evader)
        pursuers: list((int,int))
            list of coordinates of pursuers in the game (except the player's robots, if he is pursuer)
        """
        self.next_robots = self.robots[:]

        # if there are not precalculated values for policy
        if not self.loaded_policy:
            policy_file = Path("policies/" + self.game_name + ".policy")
            ###################################################
            # if there is policy file, load it...
            ###################################################
            if False:#policy_file.is_file():
                # load the strategy file
                self.loaded_policy = pickle.load(open(policy_file, 'rb'))
            ###################################################
            # ...else calculate the policy
            ###################################################
            else:
                # START OF MY CODE T4C #

                # i2c is a mapping from integer indexes to passable coordinates in the map (dict)
                # c2i is mapping from coordinates to indexes (dict)
                # Policies: dictionary, example: key=(1,0,6), value=([6,0,2],[0,0,1])
                # key in the format of (c2i[evader_position], c2i[pursuer_position], c2i[pursuer_position])
                # state (value): [6,0,2] is a list of possible c2i actions, [0,0,1] is probability if each action
                # size of array [6,0,2] depends on the number of passable coordinates
                # values = v(s) in pseudocode
                # N: number of passable coordinates in the map

                # INITIALIZATION
                mapping_i2c, mapping_c2i, N = self.init_values(gridmap)
                values = np.zeros((N, N, N))
                policy_p = {}
                policy_e = {}
                last_run = False
                converged = False
                iteration=0
                epsilon=0.0001

                # WAIT FOR CONVERGENCE (takes a while)
                while not last_run:
                    """
                    Runs until we converge (epsilon(=0.0001))
                    """
                    print("Iteration: ",iteration)
                    iteration+=1
                    if converged: last_run = True
                    converged = True
                    for E_1 in range(N):
                        for P_1 in range(N):
                            for P_2 in range(N):  

                                ngbs_E_1 = gridmap.neighbors4(mapping_i2c[E_1])
                                ngbs_P_1 = gridmap.neighbors4(mapping_i2c[P_1])
                                ngbs_P_2 = gridmap.neighbors4(mapping_i2c[P_2])

                                Q = np.zeros([len(ngbs_E_1), len(ngbs_P_1)*len(ngbs_P_2)])
                                a_e_index = -1
                                
                                for ngb_E_1 in ngbs_E_1:
                                    a_e_index+=1
                                    a_p_index = -1
                                    for ngb_P_1 in ngbs_P_1:
                                        for ngb_P_2 in ngbs_P_2:
                                            a_p_index+=1
                                            a_e = mapping_c2i[ngb_E_1]
                                            a_p1 = mapping_c2i[ngb_P_1]
                                            a_p2 = mapping_c2i[ngb_P_2]
                                            if a_e == a_p1 or a_e == a_p2 or E_1 == P_1 or E_1 == P_2 \
                                                or a_e == P_1 and a_p1 == E_1 or a_e == P_2 and a_p2 == E_1: # same action or swapping
                                                Q[a_e_index, a_p_index] = 1  #end of the game
                                            else:
                                                Q[a_e_index, a_p_index] = 0.95*values[a_e, a_p1, a_p2] #continue
                                if not last_run:
                                    val, _ = compute_nash(Q.transpose())    # wait for last run
                                    if abs(val - values[E_1, P_1, P_2]) >= epsilon: 
                                        converged = False
                                    values[E_1, P_1, P_2] = val
                                else:
                                    _, x = compute_nash(-1*Q)               # minimize
                                    _, y = compute_nash(Q.transpose())      # maximize
                                    actions_e = [mapping_c2i[ngb_E_1] for ngb_E_1 in ngbs_E_1]
                                    actions_p = []
                                    for ngb_P_1 in ngbs_P_1:
                                        for ngb_P_2 in ngbs_P_2:
                                            actions_p.append((mapping_c2i[ngb_P_1], mapping_c2i[ngb_P_2]))
                                    policy_e[(E_1, P_1, P_2)] = (actions_e, x)
                                    policy_p[(E_1, P_1, P_2)] = (actions_p, y)

                
                # END OF MY CODE T4C
                #values, policy_e, policy_p, mapping_i2c, mapping_c2i = self.compute_random_policy(gridmap)

                self.loaded_policy = (values, policy_e, policy_p, mapping_i2c, mapping_c2i)

                pickle.dump(self.loaded_policy, open(policy_file, 'wb'))

        values, policy_e, policy_p, mapping_i2c, mapping_c2i = self.loaded_policy

        if self.role == PURSUER:
            state = (mapping_c2i[evaders[0]], mapping_c2i[self.robots[0]], mapping_c2i[self.robots[1]])
        else:
            state = (mapping_c2i[self.robots[0]], mapping_c2i[pursuers[0]], mapping_c2i[pursuers[1]])

        if self.role == PURSUER:
            action_index = np.random.choice(tuple(range(len(policy_p[state][0]))), p=policy_p[state][1])
            action = policy_p[state][0][action_index]
            self.next_robots[0] = mapping_i2c[action[0]]
            self.next_robots[1] = mapping_i2c[action[1]]
        else:
            action_index = np.random.choice(tuple(range(len(policy_e[state][0]))), p=policy_e[state][1])
            action = policy_e[state][0][action_index]
            self.next_robots[0] = mapping_i2c[action]