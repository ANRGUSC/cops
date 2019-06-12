import matplotlib.pyplot as plt
import matplotlib.animation as animation

from graph_connectivity.clustering import *

#===ANIMATE=====================================================================

def animate_solution(cp, ANIM_STEP=30, filename='animation.mp4', labels=False):

    # Initiate plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis('off')

    #Setup position dictionary for node positions
    dict_pos = {n: (cp.graph.nodes[n]['x'], cp.graph.nodes[n]['y']) for n in cp.graph}

    # Build dictionary robot,time -> position
    traj_x = {(r,t): np.array([cp.graph.nodes[v]['x'], cp.graph.nodes[v]['y']])
              for (r,t), v in cp.trajectories.items()}

    # FIXED STUFF
    cluster_colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(cp.subgraphs)))
    cluster_color_dict = {c : cluster_colors[i] for i, c in enumerate(cp.subgraphs)}

    frontier_colors = {}
    if 'frontiers' in cp.graph.nodes[0]:
        frontiers = [v for v in cp.graph if cp.graph.nodes[v]['frontiers'] != 0]
        for v in frontiers:
            for c in cp.subgraphs:
                if v in cp.subgraphs[c]:
                    frontier_colors[v] = cluster_color_dict[c]
        fcolors = []
        for v in frontiers:
            fcolors.append(frontier_colors[v])
    else:
        frontiers = []
    nx.draw_networkx_nodes(cp.graph, dict_pos, ax=ax, nodelist = frontiers,
                           node_shape = "D", node_color = fcolors, edgecolors='black',
                           linewidths=1.0, alpha=0.5)

    nodes = []
    node_colors = {}
    for c in cp.subgraphs:
        for v in cp.subgraphs[c]:
            if v not in frontiers:
                nodes.append(v)
                node_colors[v] = cluster_color_dict[c]
    colors = []
    for v in nodes:
        colors.append(node_colors[v])

    nx.draw_networkx_nodes(cp.graph, dict_pos, ax=ax, nodelist = nodes,
                           node_color=colors, edgecolors='black', linewidths=1.0, alpha=0.5)
    nx.draw_networkx_edges(cp.graph, dict_pos, ax=ax, edgelist=list(cp.graph.tran_edges()),
                           connectionstyle='arc', edge_color='black')

    if labels:
        nx.draw_networkx_labels(cp.graph, dict_pos)

    # VARIABLE STUFF
    # connectivity edges
    coll_cedge = nx.draw_networkx_edges(cp.graph, dict_pos, ax=ax, edgelist=list(cp.graph.conn_edges()),
                                        edge_color='black')
    if coll_cedge is not None:
        for cedge in coll_cedge:
            cedge.set_connectionstyle("arc3,rad=0.25")
            cedge.set_linestyle('dashed')

    # robot nodes
    pos = np.array([traj_x[(r, 0)] for r in cp.graph.agents])
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

        # Update connectivity edge colors if there is flow information
        for i, (v1, v2) in enumerate(cp.graph.conn_edges()):
            coll_cedge[i].set_color('black')
            col_list = [colors[b_r] for b, b_r in enumerate(cp.graph.agents)
                        if (b_r, v1, v2, t) in cp.conn]
            if len(col_list):
                coll_cedge[i].set_color(col_list[int(10 * alpha) % len(col_list)])

        # Update robot node and label positions
        pos = (1-alpha) * np.array([traj_x[(r, min(cp.T, t))] for r in cp.graph.agents]) \
              + alpha * np.array([traj_x[(r, min(cp.T, t+1))] for r in cp.graph.agents])

        coll_rpos.set_offsets(pos)
        for i in range(len(cp.graph.agents)):
            coll_text[i].set_x(pos[i, 0])
            coll_text[i].set_y(pos[i, 1])

    ani = animation.FuncAnimation(fig, animate, range((cp.T+2) * ANIM_STEP), blit=False)

    writer = animation.writers['ffmpeg'](fps = 0.5*ANIM_STEP)
    ani.save(filename, writer=writer,dpi=100)
