import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from itertools import product

from graph_connectivity.problem import *
from graph_connectivity.explore_problem import *
from copy import deepcopy

def animate_problem_sequence(graph, problem_list, ANIM_STEP=30, filename='animation.mp4', labels=False):

    problem_time = [problem.T for problem in problem_list]
    total_time = sum(problem_time)

    # Initiate plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis('off')

    problem = problem_list[0]

    #Setup position dictionary for node positions
    dict_pos = {n: (graph.nodes[n]['x'], graph.nodes[n]['y']) for n in graph}

    # Build dictionary robot,time -> position
    traj_x = {(r,t): np.array([problem.graph.nodes[v]['x'], problem.graph.nodes[v]['y']])
              for (r,t), v in problem.trajectories.items()}


    #plot unknown graph in background
    node_color = np.full(len(graph.nodes), 'grey')
    nx.draw_networkx_nodes(graph, dict_pos, ax=ax, node_color='white', edgecolors='grey', linewidths=1.0)
    coll_tedge = nx.draw_networkx_edges(graph, dict_pos, ax=ax, edgelist=list(graph.tran_edges()), connectionstyle='arc', edge_color='grey')

    if labels:
        nx.draw_networkx_labels(graph, dict_pos, font_color='grey')

    # connectivity edges
    coll_cedge = nx.draw_networkx_edges(graph, dict_pos, ax=ax, edgelist=list(graph.conn_edges()),
                                        edge_color='grey')
    if coll_cedge is not None:
        for cedge in coll_cedge:
            cedge.set_connectionstyle("arc3,rad=0.25")
            cedge.set_linestyle('dashed')


    # robot nodes
    pos = np.array([traj_x[(r, 0)] for r in problem.graph.agents])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(problem.graph.agents)))
    coll_rpos = ax.scatter(pos[:,0], pos[:,1], s=140, marker='o',
                           c=colors, zorder=5, alpha=0.7,
                           linewidths=2, edgecolors='black')

    # robot labels
    coll_text = [ax.text(pos[i,0], pos[i,1], str(r),
                         horizontalalignment='center',
                         verticalalignment='center',
                         zorder=10, size=8, color='k',
                         family='sans-serif', weight='bold', alpha=1.0)
                 for i, r in enumerate(graph.agents)]


    if 'frontiers' in problem.graph.nodes[0]:
        frontiers = [v for v in problem.graph if problem.graph.nodes[v]['frontiers'] != 0]
    else:
        frontiers = []
    if hasattr(problem, 'reward_dict'):
        if problem.reward_dict != None:
            reward_nodes = [v for v in problem.reward_dict]
        else:
            reward_nodes = []
    else:
        reward_nodes = []
    nx.draw_networkx_nodes(graph, dict_pos, ax=ax, nodelist = list(set(frontiers) | set(reward_nodes)),
                           node_color='green', edgecolors='black', linewidths=1.0)


    def animate(time_idx):
        start_time = 0
        for problem_idx in range(len(problem_list)):
            if (start_time + problem_time[problem_idx] + 2) * ANIM_STEP <= time_idx:
                start_time += problem_time[problem_idx] + 2
            else:
                problem = problem_list[problem_idx]
                problem_number = problem_idx
                break
        i = time_idx - start_time * ANIM_STEP
        t = int(i / ANIM_STEP)
        anim_idx = i % ANIM_STEP
        alpha = anim_idx / ANIM_STEP

        if i == 0:
            print(str('Animating: ' + str(type(problem))))

        if isinstance(problem, ConnectivityProblem):
            for n in problem.graph.nodes:
                if problem.graph.nodes[n]['known']:
                    node_color[n] = 'k'
            coll_node = nx.draw_networkx_nodes(graph, dict_pos, ax=ax, nodelist = graph.nodes,
                               node_color='white', edgecolors=node_color, linewidths=1.0)
            if 'frontiers' in problem.graph.nodes[0]:
                frontiers = [v for v in problem.graph if problem.graph.nodes[v]['frontiers'] != 0]
            else:
                frontiers = []
            if hasattr(problem, 'reward_dict'):
                if problem.reward_dict != None:
                    reward_nodes = [v for v in problem.reward_dict]
                else:
                    reward_nodes = []
            else:
                reward_nodes = []
            nx.draw_networkx_nodes(graph, dict_pos, ax=ax, nodelist = list(set(frontiers) | set(reward_nodes)),
                                   node_color='green', edgecolors='black', linewidths=1.0)

            # Update connectivity edge colors if there is flow information
            if 'fbar' in problem.vars and 'mbar' in problem.vars:
                for i, (v1, v2) in enumerate(graph.conn_edges()):
                    if (v1, v2) in problem.graph.conn_edges():
                        coll_cedge[i].set_color('black')
                        col_list = [colors[b_r] for b, b_r in enumerate(problem.min_src_snk)
                                    if problem.solution['x'][problem.get_fbar_idx(b, v1, v2, min(problem.T, t))] > 0.5]
                        if problem.solution['x'][problem.get_mbar_idx(v1, v2, min(problem.T, t))] > 0.5:
                             col_list.append(colors[problem.master])
                        if len(col_list):
                            coll_cedge[i].set_color(col_list[int(10 * alpha) % len(col_list)])


        if isinstance(problem, ExplorationProblem):
            g = problem.graph_list[min(problem.T, t)]
            #Update node 'known' color
            for n in g.nodes:
                if g.nodes[n]['known']:
                    node_color[n] = 'k'
            coll_node = nx.draw_networkx_nodes(graph, dict_pos, ax=ax, nodelist = graph.nodes,
                               node_color='white', edgecolors=node_color, linewidths=1.0)

            for n in g.nodes:
                if g.nodes[n]['known']:
                    nx.draw_networkx_labels(graph, dict_pos, labels={n:n}, font_color='black')

            #Update transition-edge 'known' color
            for i, (v1, v2) in enumerate(g.tran_edges()):
                if g.nodes[v1]['known'] == True and g.nodes[v2]['known'] == True:
                    coll_tedge[i].set_color('black')

            #Update connectivity-edge 'known' color
            for i, (v1, v2) in enumerate(g.conn_edges()):
                if g.nodes[v1]['known'] == True and g.nodes[v2]['known'] == True:
                    coll_cedge[i].set_color('black')

            last_graph = problem_list[problem_number - 1].graph

            if 'frontiers' in last_graph.nodes[0]:
                frontiers = [v for v in last_graph if last_graph.nodes[v]['frontiers'] != 0]
            else:
                frontiers = []
            if hasattr(problem, 'reward_dict'):
                if problem.reward_dict != None:
                    reward_nodes = [v for v in problem.reward_dict]
                else:
                    reward_nodes = []
            else:
                reward_nodes = []
            nx.draw_networkx_nodes(graph, dict_pos, ax=ax, nodelist = list(set(frontiers) | set(reward_nodes)),
                                   node_color='green', edgecolors='black', linewidths=1.0)

            # Update connectivity edge colors if there is flow information
            if 'fbar' in problem.vars:
                for i, (v1, v2) in enumerate(g.conn_edges()):
                    if g.nodes[v1]['known'] == True and g.nodes[v2]['known'] == True:
                        coll_cedge[i].set_color('black')
                    col_list = [colors[r] for fr, r in problem.frontier_robot_dict.items()
                                if problem.fbar[problem.get_fbar_idx(fr, v1, v2, min(problem.T, t))] > 0.5]
                    if len(col_list):
                        coll_cedge[i].set_color(col_list[int(10 * alpha) % len(col_list)])

        # Build dictionary robot,time -> position
        traj_x = {(r,t): np.array([problem.graph.nodes[v]['x'], problem.graph.nodes[v]['y']])
                  for (r,t), v in problem.trajectories.items()}

        # Update robot node and label positions
        pos = (1-alpha) * np.array([traj_x[(r, min(problem.T, t))] for r in problem.graph.agents]) \
              + alpha * np.array([traj_x[(r, min(problem.T, t+1))] for r in problem.graph.agents])

        coll_rpos.set_offsets(pos)
        for i in range(len(problem.graph.agents)):
            coll_text[i].set_x(pos[i, 0])
            coll_text[i].set_y(pos[i, 1])




    ani = animation.FuncAnimation(fig, animate, range((total_time + 2 * len(problem_list) ) * ANIM_STEP), blit=False)

    writer = animation.writers['ffmpeg'](fps = 0.5*ANIM_STEP)
    ani.save(filename, writer=writer,dpi=100)
