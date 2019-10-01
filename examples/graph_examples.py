from cops.graph import Graph


def get_linear_graph(n):
    G = Graph()
    G.add_transition_path(list(range(n)))
    G.add_connectivity_path(list(range(n)))
    G.set_node_positions({i: (0, i) for i in range(n)})

    return G


def get_medium_graph():

    # Define a connectivity graph
    G = Graph()
    # Add edges to graph
    connectivity_edges = [0, 1, 2, 3, 4, 5]
    transition_edges = [0, 1, 2, 3, 4, 5]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [4, 6, 7, 8]
    transition_edges = [4, 6, 7, 8]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [1, 9, 10, 11, 12, 13]
    transition_edges = [1, 9, 10, 11, 12, 13]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [9, 14, 15, 12]
    transition_edges = [9, 14, 15, 12]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [1, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    transition_edges = [1, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [8, 10]
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [15, 24]
    G.add_connectivity_path(connectivity_edges)
    # Set node positions
    node_positions = {
        0: (0, 0),
        1: (0, 1),
        2: (-2, 1),
        3: (-2, 2),
        4: (-3, 2),
        5: (-3, 1),
        6: (-3, 3),
        7: (-3.5, 3.5),
        8: (-2.5, 3.5),
        9: (0, 3),
        10: (-1.8, 3),
        11: (-1.6, 4),
        12: (0, 4),
        13: (0, 5),
        14: (1, 3),
        15: (1, 4),
        16: (1.5, 1),
        17: (2.5, 1.3),
        18: (4, 1.3),
        19: (5, 1.3),
        20: (5, 2),
        21: (4, 3),
        22: (5, 3),
        23: (5, 4),
        24: (3.5, 4),
    }

    G.set_node_positions(node_positions)

    return G


def get_huge_graph():

    G = Graph()

    # Add edges to graph
    connectivity_edges = [0, 1, 2, 3, 4, 5]
    transition_edges = [0, 1, 2, 3, 4, 5]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [4, 6, 7, 8]
    transition_edges = [4, 6, 7, 8]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [1, 9, 10, 11, 12, 13]
    transition_edges = [1, 9, 10, 11, 12, 13]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [9, 14, 15, 12]
    transition_edges = [9, 14, 15, 12]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [1, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    transition_edges = [1, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [8, 10]
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [15, 24]
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [11, 25, 26, 27, 28, 29, 30, 31]
    transition_edges = [11, 25, 26, 27, 28, 29, 30, 31]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [29, 32, 33, 34, 35]
    transition_edges = [29, 32, 33, 34, 35]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [22, 36, 37, 38, 39, 40, 41]
    transition_edges = [22, 36, 37, 38, 39, 40, 41]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [39, 42, 43, 44, 45]
    transition_edges = [39, 42, 43, 44, 45]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [5, 46, 47, 48, 49]
    transition_edges = [5, 46, 47, 48, 49]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [15, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    transition_edges = [15, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [55, 61, 62, 63, 64, 65, 66, 67]
    transition_edges = [55, 61, 62, 63, 64, 65, 66, 67]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [46, 68, 69, 70, 71, 72, 73, 74, 75, 76, 33]
    transition_edges = [46, 68, 69, 70, 71, 72, 73, 74, 75, 76, 33]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [75, 77, 78, 27]
    transition_edges = [75, 77, 78, 27]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [76, 79, 80, 81, 82, 83, 84, 85, 86, 87]
    transition_edges = [76, 79, 80, 81, 82, 83, 84, 85, 86, 87]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [36, 88, 89, 90, 91, 92, 93, 94, 95]
    transition_edges = [36, 88, 89, 90, 91, 92, 93, 94, 95]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)
    connectivity_edges = [24, 96, 97, 98, 99]
    transition_edges = [24, 96, 97, 98, 99]
    G.add_transition_path(transition_edges)
    G.add_connectivity_path(connectivity_edges)

    # Set node positions
    node_positions = {
        0: (0, 0),
        1: (0, 1),
        2: (-2, 1),
        3: (-2, 2),
        4: (-3, 2),
        5: (-3, 1),
        6: (-3, 3),
        7: (-3.5, 3.5),
        8: (-2.5, 3.5),
        9: (0, 3),
        10: (-1.8, 3),
        11: (-1.6, 4),
        12: (0, 4),
        13: (0, 5),
        14: (1, 3),
        15: (1, 4),
        16: (1.5, 1),
        17: (2.5, 1.3),
        18: (4, 1.3),
        19: (5, 1.3),
        20: (5, 2),
        21: (4, 3),
        22: (5, 3),
        23: (5, 4),
        24: (3.5, 4),
        25: (-1.6, 5),
        26: (-1.7, 6),
        27: (-2, 6.5),
        28: (-2, 7.5),
        29: (-1.5, 8.5),
        30: (-1.5, 10),
        31: (-1.5, 11),
        32: (-2.5, 8.8),
        33: (-3.5, 9),
        34: (-4, 10),
        35: (-4.5, 11),
        36: (6, 3),
        37: (6.5, 3.8),
        38: (7.2, 4.5),
        39: (7.5, 5.5),
        40: (7, 6),
        41: (7, 7),
        42: (8.5, 6),
        43: (9.5, 6.5),
        44: (10, 7),
        45: (11, 7),
        46: (-4, 1),
        47: (-4.5, 2),
        48: (-5, 3),
        49: (-5, 4),
        50: (1, 5),
        51: (1.2, 6),
        52: (1.5, 7),
        53: (2, 8),
        54: (2, 9),
        55: (3, 10),
        56: (4, 10.5),
        57: (5, 11),
        58: (6, 11),
        59: (6.5, 12),
        60: (7, 13),
        61: (2.5, 11),
        62: (2, 12),
        63: (2, 13),
        64: (1, 14),
        65: (0, 15),
        66: (-1, 15.5),
        67: (-2.5, 15),
        68: (-5, 1),
        69: (-6, 1),
        70: (-7, 2),
        71: (-7, 3),
        72: (-7.5, 4),
        73: (-7.5, 5),
        74: (-6, 6),
        75: (-5, 7),
        76: (-4, 8),
        77: (-4, 6),
        78: (-3, 6),
        79: (-5, 8),
        80: (-5, 9.2),
        81: (-5.5, 10),
        82: (-6, 11),
        83: (-7, 11),
        84: (-7, 12),
        85: (-6.5, 13),
        86: (-6, 14),
        87: (-7, 15),
        88: (7, 3),
        89: (8, 3),
        90: (9, 4),
        91: (9.5, 5),
        92: (10, 5),
        93: (11, 6),
        94: (12, 7),
        95: (13, 8),
        96: (3.5, 5),
        97: (4, 6),
        98: (4, 7),
        99: (4.5, 8),
    }
    G.set_node_positions(node_positions)

    # Set small nodes
    small_nodes = [v for v in node_positions]
    small_nodes.remove(0)
    G.set_small_node(small_nodes)

    return G
