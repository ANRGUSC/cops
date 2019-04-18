import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from itertools import product

def animate_solution(cp):
    G = cp.graph
    ANIM_STEP = 30

    # Initiate plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis('off')

    #Setup position dictionary for node positions
    position_dictionary = {n: (G.nodes[n]['x'],G.nodes[n]['y']) for n in G}

    # Build dictionary robot,time -> position
    trajectories = {}
    for r, v, t in product(cp.graph.agents, cp.graph.nodes, range(cp.T+1)):
        if cp.solution['x'][cp.get_z_idx(r, v, t)] > 0.5:
            trajectories[(r,t)] = np.array([G.nodes[v]['x'], G.nodes[v]['y']])

    #Setup lists with connectivity edges and transition edges separately
    connectivity_edgelist = [(v1, v2) for (v1, v2, etype) in G.edges(data='type') if etype == "connectivity"]
    transition_edgelist = [(v1, v2) for (v1, v2, etype) in G.edges(data='type') if etype != "connectivity"]

    # FIXED STUFF
    nx.draw_networkx_nodes(G, position_dictionary, ax=ax, 
                           node_color='white', edgecolors='black', linewidths=1.0)
    nx.draw_networkx_edges(G, position_dictionary, ax=ax, edgelist=transition_edgelist,
                           connectionstyle='arc', edge_color='black')

    # VARIABLE STUFF
    # connectivity edges
    coll_cedge = nx.draw_networkx_edges(G, position_dictionary, ax=ax, edgelist=connectivity_edgelist, 
                                        edge_color='black')
    for cedge in coll_cedge:
        cedge.set_connectionstyle("arc3,rad=0.25")
        cedge.set_linestyle('dashed')

    # robot nodes
    pos = np.array([trajectories[(r, 0)] for r in cp.graph.agents])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(cp.graph.agents)))
    coll_rpos = ax.scatter(pos[:,0], pos[:,1], s=140, marker='o', 
                           c=colors, zorder=5, alpha=0.7,
                           linewidths=2, edgecolors='black')

    # robot labels
    coll_text = [ax.text(pos[i,0], pos[i,1], str(r), 
                         horizontalalignment='center', 
                         verticalalignment='center',
                         zorder=10, size=8, color='k', 
                         family='sans-serif', weight='bold', alpha=1.0)
                 for i, r in enumerate(cp.graph.agents)]

    def animate(i):
        t = int(i / ANIM_STEP)
        anim_idx = i % ANIM_STEP
        alpha = anim_idx / ANIM_STEP

        # Update connectivity edge colors
        for i, (v1, v2) in enumerate(connectivity_edgelist):
            coll_cedge[i].set_color('black')
            col_list = [colors[b] for b in cp.b 
                        if cp.solution['x'][cp.get_fbar_idx(b, v1, v2, min(cp.T, t))] > 0.5]
            if len(col_list):
                coll_cedge[i].set_color(col_list[int(10 * alpha) % len(col_list)])

        # Update robot node and label positions
        pos = (1-alpha) * np.array([trajectories[(r, min(cp.T, t))] for r in cp.graph.agents]) \
              + alpha * np.array([trajectories[(r, min(cp.T, t+1))] for r in cp.graph.agents])
        
        coll_rpos.set_offsets(pos)
        for i in range(len(cp.graph.agents)):
            coll_text[i].set_x(pos[i, 0])
            coll_text[i].set_y(pos[i, 1])

    ani = animation.FuncAnimation(fig, animate, range((cp.T+2) * ANIM_STEP), blit=False)

    writer = animation.writers['ffmpeg'](fps = 0.5*ANIM_STEP)
    ani.save('test.mp4',writer=writer,dpi=100)
