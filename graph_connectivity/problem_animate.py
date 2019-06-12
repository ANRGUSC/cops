import matplotlib.pyplot as plt
import matplotlib.animation as animation
from networkx.drawing.nx_agraph import to_agraph

from graph_connectivity.problem import *

##VISUALIZATION FUNCTIONS##

def plot_solution(self):
    if self.solution is None:
        print("No solution found: call solve() to obtain solution")
    else:
        g = self.graph
        #create time-augmented graph G
        G = Graph()
        #space between timesteps
        space = 2

        #loop of graph and add nodes to time-augmented graph
        for t in range(self.T+1):
            for n in g.nodes:
                id = self.get_time_augmented_id(n, t)
                G.add_node(id)
                G.nodes[id]['x'] = g.nodes[n]['x'] + space*t
                G.nodes[id]['y'] = g.nodes[n]['y']
                G.nodes[id]['agents'] = []

        #loop of graph and add edges to time-augmented graph
        i = 0
        for n in g.nodes:
            for t in range(self.T+1):
                for edge in g.out_edges(n, data = True):
                    if edge[2]['type'] == 'transition' and t < self.T:
                        out_id = self.get_time_augmented_id(n, t)
                        in_id = self.get_time_augmented_id(edge[1], t+1)
                        G.add_edge(out_id , in_id, type='transition')
                    elif edge[2]['type'] == 'connectivity':
                        out_id = self.get_time_augmented_id(n, t)
                        in_id = self.get_time_augmented_id(edge[1], t)
                        G.add_edge(out_id , in_id, type='connectivity')


        #set 'pos' attribute for accurate position in plot (use ! in end for accurate positioning)
        for n in G:
            G.nodes[n]['pos'] =  ",".join(map(str,[G.nodes[n]['x'],G.nodes[n]['y']])) + '!'

        #Set general edge-attributes
        G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}

        #Set individual attributes based on solution
        for (r,t), v in self.trajectories.items():
            G.nodes[self.get_time_augmented_id(v, t)]['agents'].append(r)

        for v in G.nodes:
            G.nodes[v]['number_of_agents'] = len(G.nodes[v]['agents'])
            G.nodes[v]['label'] = " ".join(map(str,(G.nodes[v]['agents'])))

        for n in G:
            if G.nodes[n]['number_of_agents']!=0:
                G.nodes[n]['color'] = 'black'
                G.nodes[n]['fillcolor'] = 'red'
                G.nodes[n]['style'] = 'filled'
            else:
                G.nodes[n]['color'] = 'black'
                G.nodes[n]['fillcolor'] = 'white'
                G.nodes[n]['style'] = 'filled'
            for nbr in G[n]:
                for edge in G[n][nbr]:
                    if G[n][nbr][edge]['type']=='connectivity':
                        if len(G.nodes[n]['agents']) != 0 and len(G.nodes[nbr]['agents'])!= 0:
                            G[n][nbr][edge]['color']='black'
                            G[n][nbr][edge]['style']='dashed'
                        else:
                            G[n][nbr][edge]['color']='grey'
                            G[n][nbr][edge]['style']='dashed'
                    else:
                        G[n][nbr][edge]['color']='grey'
                        G[n][nbr][edge]['style']='solid'
        for n in G:
            for nbr in G[n]:
                for edge in G[n][nbr]:
                    if G[n][nbr][edge]['type']=='transition':
                        for a in G.nodes[n]['agents']:
                            if a in G.nodes[nbr]['agents']:
                                G[n][nbr][edge]['color']='black'
                                G[n][nbr][edge]['style']='solid'


        #Plot/save graph
        A = to_agraph(G)
        A.layout()
        A.draw('solution.png')

def animate_solution(prob, full_graph = None, ANIM_STEP=30, filename='animation.mp4', labels=False):

    # Initiate plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis('off')

    #if we have full graph, plot it in background
    if full_graph:
            #Setup position dictionary for node positions
            dict_pos = {n: (full_graph.nodes[n]['x'], full_graph.nodes[n]['y']) for n in full_graph}

            # FIXED STUFF
            nx.draw_networkx_nodes(full_graph, dict_pos, ax=ax,
                                   node_color='white', edgecolors='grey', linewidths=1.0)
            nx.draw_networkx_edges(full_graph, dict_pos, ax=ax, edgelist=list(full_graph.tran_edges()),
                                   connectionstyle='arc', edge_color='grey')

            if labels:
                nx.draw_networkx_labels(full_graph, dict_pos, font_color='grey')

            # VARIABLE STUFF
            # connectivity edges
            coll_cedge = nx.draw_networkx_edges(full_graph, dict_pos, ax=ax, edgelist=list(full_graph.conn_edges()),
                                                edge_color='grey')
            if coll_cedge is not None:
                for cedge in coll_cedge:
                    cedge.set_connectionstyle("arc3,rad=0.25")
                    cedge.set_linestyle('dashed')

    #Plot IRM

    #Setup position dictionary for node positions
    dict_pos = {n: (prob.graph.nodes[n]['x'], prob.graph.nodes[n]['y']) for n in prob.graph}

    # Build dictionary robot,time -> position
    traj_x = {(r,t): np.array([prob.graph.nodes[v]['x'], prob.graph.nodes[v]['y']])
              for (r,t), v in prob.trajectories.items()}

    # FIXED STUFF
    nx.draw_networkx_nodes(prob.graph, dict_pos, ax=ax,
                           node_color='white', edgecolors='black', linewidths=1.0)
    nx.draw_networkx_edges(prob.graph, dict_pos, ax=ax, edgelist=list(prob.graph.tran_edges()),
                           connectionstyle='arc', edge_color='black')

    if 'frontiers' in prob.graph.nodes[0]:
        frontiers = [v for v in prob.graph if prob.graph.nodes[v]['frontiers'] != 0]
    else:
        frontiers = []
    if prob.reward_dict != None:
        reward_nodes = [v for v in prob.reward_dict]
    else:
        reward_nodes = []
    nx.draw_networkx_nodes(prob.graph, dict_pos, ax=ax, nodelist = list(set(frontiers) | set(reward_nodes)),
                           node_color='green', edgecolors='black', linewidths=1.0)

    if labels:
        nx.draw_networkx_labels(prob.graph, dict_pos)

    # VARIABLE STUFF
    # connectivity edges
    coll_cedge = nx.draw_networkx_edges(prob.graph, dict_pos, ax=ax, edgelist=list(prob.graph.conn_edges()),
                                        edge_color='black')
    if coll_cedge is not None:
        for cedge in coll_cedge:
            cedge.set_connectionstyle("arc3,rad=0.25")
            cedge.set_linestyle('dashed')

    # robot nodes
    pos = np.array([traj_x[(r, 0)] for r in prob.graph.agents])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(prob.graph.agents)))
    coll_rpos = ax.scatter(pos[:,0], pos[:,1], s=140, marker='o',
                           c=colors, zorder=5, alpha=0.7,
                           linewidths=2, edgecolors='black')

    # robot labels
    coll_text = [ax.text(pos[i,0], pos[i,1], str(r),
                         horizontalalignment='center',
                         verticalalignment='center',
                         zorder=10, size=8, color='k',
                         family='sans-serif', weight='bold', alpha=1.0)
                 for i, r in enumerate(prob.graph.agents)]

    def animate(i):
        t = int(i / ANIM_STEP)
        anim_idx = i % ANIM_STEP
        alpha = anim_idx / ANIM_STEP

        # Update connectivity edge colors if there is flow information
        if 'fbar' in prob.vars:
            for i, (v1, v2) in enumerate(prob.graph.conn_edges()):
                coll_cedge[i].set_color('black')
                col_list = [colors[b_r] for b, b_r in enumerate(prob.min_src_snk)
                            if prob.solution['x'][prob.get_fbar_idx(b, v1, v2, min(prob.T, t))] > 0.5]
                if len(col_list):
                    coll_cedge[i].set_color(col_list[int(10 * alpha) % len(col_list)])

        # Update robot node and label positions
        pos = (1-alpha) * np.array([traj_x[(r, min(prob.T, t))] for r in prob.graph.agents]) \
              + alpha * np.array([traj_x[(r, min(prob.T, t+1))] for r in prob.graph.agents])

        coll_rpos.set_offsets(pos)
        for i in range(len(prob.graph.agents)):
            coll_text[i].set_x(pos[i, 0])
            coll_text[i].set_y(pos[i, 1])

    ani = animation.FuncAnimation(fig, animate, range((prob.T+2) * ANIM_STEP), blit=False)

    writer = animation.writers['ffmpeg'](fps = 0.5*ANIM_STEP)
    ani.save(filename, writer=writer,dpi=100)
