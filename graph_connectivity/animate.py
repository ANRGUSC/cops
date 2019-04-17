import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from itertools import product

def animate_solution(cp):
    G = cp.graph
    ANIM_STEP = 30

    # Initiate plot
    fig = plt.figure(figsize=(10,10))
    plt.box(on=None)

    #Setup position dictionary for node positions
    position_dictionary = {n: (G.nodes[n]['x'],G.nodes[n]['y']) for n in G}

    #Setup lists with connectivity edges and transition edges separately
    connectivity_edgelist =[]
    transition_edgelist =[]
    for (v1,v2,type) in G.edges(data='type'):
        if type == 'connectivity':
            connectivity_edgelist.append((v1,v2))
        else:
            transition_edgelist.append((v1,v2))

    #Init agent animation
    agent_nodes = []
    agent_positions = {}
    agent_labels = {}
    for r, v in product(cp.graph.agents, cp.graph.nodes):
            if cp.solution['x'][cp.get_z_idx(r, v, 0)] != 0:
                agent_id = 'r' + str(r)
                agent_labels[agent_id] = str(r)
                agent_nodes.append(agent_id)
                x_pos = cp.graph.nodes[v]['x']
                y_pos = cp.graph.nodes[v]['y']
                agent_positions[agent_id] = (x_pos , y_pos)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    rads = [x * 0.1 + 0.1 for x in range(cp.num_b)]

    def animate(i):
        fig.clear()
        plt.box(on=None)
        t = int(i / ANIM_STEP)
        anim_idx = i % ANIM_STEP

        nx.draw_networkx_nodes(G, position_dictionary, node_color = 'white', edgecolors = 'black', linewidths = 1.0)
        nx.draw_networkx_edges(G, position_dictionary, edgelist = transition_edgelist, connectionstyle = 'arc', edge_color = 'black')
        c_edges = nx.draw_networkx_edges(G, position_dictionary, edgelist = connectivity_edgelist, connectionstyle = 'arc3, rad = 0.1', edge_color = 'black')
        for patch in c_edges:
            patch.set_linestyle('dashed')

        #Update connectivity edge colors
        for v1, v2 in product(cp.graph.nodes, cp.graph.nodes):
                rads_idx = 0
                for b in cp.b:
                    if cp.solution['x'][cp.get_fbar_idx(b, v1, v2, t)] != 0:
                        c_edge = nx.draw_networkx_edges(G, position_dictionary, edgelist = [(v1,v2)], connectionstyle = 'arc3, rad = ' + str(rads[rads_idx]), edge_color = colors[b])
                        rads_idx += 1
                        for patch in c_edge:
                            patch.set_linestyle('dashed')

        #Update agent node positions
        agent_positions = {}
        for r, v1, v2 in product(cp.graph.agents, cp.graph.nodes, cp.graph.nodes):
                if cp.solution['x'][cp.get_z_idx(r, v1, t)] != 0:
                    if t < cp.T:
                        if cp.solution['x'][cp.get_z_idx(r, v2, t+1)] != 0:
                            agent_id = 'r' + str(r)
                            x_pos = cp.graph.nodes[v1]['x'] + (anim_idx / ANIM_STEP) * (cp.graph.nodes[v2]['x'] - cp.graph.nodes[v1]['x'])
                            y_pos = cp.graph.nodes[v1]['y'] + (anim_idx / ANIM_STEP) * (cp.graph.nodes[v2]['y'] - cp.graph.nodes[v1]['y'])
                            agent_positions[agent_id] = (x_pos , y_pos)
                    else:
                        agent_id = 'r' + str(r)
                        x_pos = cp.graph.nodes[v1]['x']
                        y_pos = cp.graph.nodes[v1]['y']
                        agent_positions[agent_id] = (x_pos , y_pos)
        nx.draw_networkx_nodes(G, pos = agent_positions, nodelist = agent_nodes, node_color = colors[0:len(agent_nodes)], node_size = 100)
        nx.draw_networkx_labels(G, pos = agent_positions, labels = agent_labels, font_size=8, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0)


    ani = animation.FuncAnimation(fig, animate, range((cp.T + 1) * ANIM_STEP), blit=False)
    #plt.show()

    writer = animation.writers['ffmpeg'](fps = 0.2*ANIM_STEP)
    ani.save('test.mp4',writer=writer,dpi=100)
