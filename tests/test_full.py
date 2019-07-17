from cops.problem import *
import numpy as np

def test_horiz1():
    G = Graph()
    connectivity_edges = [0,1,2,3] #directed connectivity path (one way)
    transition_edges = [0,1,2,3]   #directed transition path (one way)
    #Add edges to graph
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)

    #Set initial position of agents
    agent_positions = {0: 0, 1: 1, 2: 3}    #agent:position
    #Init agents in graphs
    G.init_agents(agent_positions)
    
    # Set up the connectivity problem
    cp = ConnectivityProblem()

    # Specify graph
    cp.graph = G
    # Specify time horizon
    cp.T = 1
    # Specify sources
    cp.src = [2]

    cp.static_agents = [2]

    print('Solving')
    cp.solve_powerset()

    # positions of robot 0
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(0,0,0)], 1)
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(0,1,1)], 1)
            
    # positions of robot 1
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(1,1,0)], 1)
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(1,2,1)], 1)
            
    # positions of robot 2 (fixed)
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(2,3,0)], 1)
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(2,3,1)], 1)

def test_horiz2():
    G = Graph()
    connectivity_edges = [0,1,2,3] #directed connectivity path (one way)
    transition_edges = [0,1,2,3]   #directed transition path (one way)
    #Add edges to graph
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)

    #Set initial position of agents
    agent_positions = {0: 0, 1: 1, 2: 3}    #agent:position
    #Init agents in graphs
    G.init_agents(agent_positions)
    
    # Set up the connectivity problem
    cp = ConnectivityProblem()

    # Specify graph
    cp.graph = G
    # Specify time horizon
    cp.T = 2
    # Specify sources
    cp.src = [2]

    cp.static_agents = [0, 2]

    print('Solving')
    cp.solve_powerset()

    # positions of robot 0
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(0,0,0)], 1)
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(0,0,1)], 1)
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(0,0,2)], 1)
            
    # positions of robot 1
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(1,1,0)], 1)
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(1,2,1)], 1)
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(1,1,2)], 1)
            
    # positions of robot 2 (fixed)
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(2,3,0)], 1)
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(2,3,1)], 1)
    np.testing.assert_equal(cp.solution['x'][cp.get_z_idx(2,3,2)], 1)            
