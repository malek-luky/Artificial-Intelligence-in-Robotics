from sensor_oracle import pair_evaluation, best_sensor
from planning_oracle import cost_function, best_plan
import numpy as np
import gurobipy as gp
from gurobipy import *

def compute_nash(matrix, only_value=False, minimize=False):
    """
    REUSED FROM T4ac-resource Player.py

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
    return [x[i].x for i in range(num_actions)]
    
    # END OF MY CODE
    #maximizer_actions = matrix.shape[0]
    #return 0, [1./maximizer_actions]*maximizer_actions

def double_oracle(gallery, epsilon=1e-6):
    """
    Method to compute optimal strategy for attacker and defender using double oracle algorithm and oracles implemented in previous steps

    Parameters
    ----------
    gallery: Gallery
        Map of the environment with information about sensors of a given size
    epsilon: double
        The distance between both player's best response values required to terminate the algorithm
    Returns
    -------
    sensor_strategy: list(double)
        Optimal strategy as a probability distribution over sensors
    sensors: list(int)
        List of sensors used as actions
    path_strategy: list(double)
        Optimal strategy as a probability distribution over paths
    paths: list(list(tuple(int, int)))
        List of all the paths used as actions
    """

    # START OF MY CODE

    # STEPS:
    # 1. Choose a random sensor and a random path
    # 2. Compute the best response of the attacker to the defender's strategy
    # 3. Compute the best response of the defender to the attacker's strategy
    # 4. If the best responses are close enough, stop. Otherwise, go to step 2

    # 1. Choose a random sensor and a random path
    sensors = [0]
    paths = [best_plan(gallery, np.zeros([gallery.y_size, gallery.x_size]))[0]]
    # we dont need to check if the item is already in a list if we use set
    # but set is not ordered, so we have to use list
    #tududumdum


    while True:

        # 1. Create the matrix [size chosen paths x chosen sensors] for fiding nash equilibrium using the
        m = np.zeros([len(paths),len(sensors)])
        for x,path in enumerate(paths):
            for y,sensor in enumerate(sensors):
                m[x,y] = pair_evaluation(gallery, path, sensor)
        # nyni mame v matici m hodnoty vsech kombinaci vybranych cest a senzoru
        
        # 2. Compute the best response of the attacker to the defender's strategy
        path_probs = compute_nash(-m)
        sensor_probs= compute_nash(m.T)
        # nyni mame vybrane pravdepodobnostni rozdeleni pro oba hrace

        # 3. Compute the best response of the defender to the attacker's strategy and vice versa
        # defender = chooses sensors
        # attacker = chooses paths
        best_attacker_response, attacker_value = best_plan(gallery, cost_function(gallery, [sensor_probs, sensors]))
        best_defender_response, defender_value = best_sensor(gallery, [path_probs, paths])

        # 4. If the best responses are close enough, stop. Otherwise, go to step 2
        if abs(attacker_value - defender_value) < epsilon:
            break
        else:
            if best_defender_response not in sensors:
                sensors.append(best_defender_response)
            if best_attacker_response not in paths:
                paths.append(best_attacker_response)

    return sensor_probs, sensors, path_probs, paths

    # END OF MY CODE