from threading import Thread
import functools
from collections import defaultdict, deque
from timeit import default_timer as timer
import sys
import os
from functools import lru_cache
from functools import wraps
from collections import namedtuple
import json
import six
from collections.abc import Iterable

# taken from https://gist.github.com/harel/9ced5ed51b97a084dec71b9595565a71
Serialized = namedtuple('Serialized', 'json')

class hash_list(list): 
    def __init__(self, *args): 
        if len(args) == 1 and isinstance(args[0], Iterable): 
            args = args[0] 
        super().__init__(args) 
         
    def __hash__(self): 
        return hash(e for e in self)

def hashable_lru(func):
    cache = lru_cache(maxsize=2048)

    def deserialise(value):
        try:
            return json.loads(value)
        except Exception:
            return value

    def func_with_serialized_params(*args, **kwargs):
        _args = tuple([deserialise(arg) for arg in args])
        _kwargs = {k: deserialise(v) for k, v in kwargs.items()}
        return func(*_args, **_kwargs)

    cached_function = cache(func_with_serialized_params)

    @wraps(func)
    def lru_decorator(*args, **kwargs):
        _args = tuple([json.dumps(arg, sort_keys=True) if type(arg) in (list, dict) else arg for arg in args])
        _kwargs = {k: json.dumps(v, sort_keys=True) if type(v) in (list, dict) else v for k, v in kwargs.items()}
        return cached_function(*_args, **_kwargs)
    lru_decorator.cache_info = cached_function.cache_info
    lru_decorator.cache_clear = cached_function.cache_clear
    return lru_decorator

def hashable_cache(cache):
    def hashable_cache_internal(func):
        def deserialize(value):
            if isinstance(value, Serialized):
                return json.loads(value.json)
            else:
                return value

        def func_with_serialized_params(*args, **kwargs):
            _args = tuple([deserialize(arg) for arg in args])
            _kwargs = {k: deserialize(v) for k, v in six.viewitems(kwargs)}
            return func(*_args, **_kwargs)

        cached_func = cache(func_with_serialized_params)

        @functools.wraps(func)
        def hashable_cached_func(*args, **kwargs):
            _args = tuple([
                Serialized(json.dumps(arg, sort_keys=True))
                if type(arg) in (list, dict) else arg
                for arg in args
            ])
            _kwargs = {
                k: Serialized(json.dumps(v, sort_keys=True))
                if type(v) in (list, dict) else v
                for k, v in kwargs.items()
            }
            return cached_func(*_args, **_kwargs)
        hashable_cached_func.cache_info = cached_func.cache_info
        hashable_cached_func.cache_clear = cached_func.cache_clear
        return hashable_cached_func

    return hashable_cache_internal


# representation based on dictionary, each key is a node and each entry corresponds to the list nodes connected by a direct arc
graphForTests = { 'A': list(['B', 'C']),
        'B': list(['C', 'D']),
        'C': list(['D']),
        'D': list(['C']),
        'E': list(['F']),
        'F': list(['C']) }

gdict = { "a" : set(["b","c"]),
                "b" : set(["a", "d"]),
                "c" : set(["a", "d"]),
                "d" : set(["e"]),
                "e" : set(["a"])
                }

def printGraph(graph):
    for k in graph:
        nodes = graph.get(k)
        if nodes != None:
            print(str(k) + " => " + str(nodes))

def find_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if start not in graph:
        return None
    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath: return newpath
    return None

@hashable_cache(lru_cache())
#@hashable_lru
def find_all_paths(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if start not in graph:
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths 

def find_shortest_path2(graph, start, end):
        dist = {start: [start]}
        q = deque(start)
        while len(q):
            at = q.popleft()
            for next in graph[at]:
                if next not in dist:
                    dist[next] = [dist[at], next]
                    q.append(next)
        return dist.get(end)

def find_shortest_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if start not in graph:
            return None
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = find_shortest_path(graph, node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest


# Function to find the shortest path between two nodes of a graph
# Taken from https://www.geeksforgeeks.org/building-an-undirected-graph-and-finding-shortest-path-using-dictionaries-in-python/
def BFS_SP(graph, start, goal):
	explored = []
	
	# Queue for traversing the 
	# graph in the BFS
	queue = [[start]]
	
	# If the desired node is 
	# reached
	if start == goal:
		print("Same Node")
		return
	
	# Loop to traverse the graph 
	# with the help of the queue
	while queue:
		path = queue.pop(0)
		node = path[-1]
		
		# Condition to check if the
		# current node is not visited
		if node not in explored:
			neighbours = graph[node]
			
			# Loop to iterate over the 
			# neighbours of the node
			for neighbour in neighbours:
				new_path = list(path)
				new_path.append(neighbour)
				queue.append(new_path)
				
				# Condition to check if the 
				# neighbour node is the goal
				if neighbour == goal:
					#print("Shortest path = ", *new_path)
					return new_path
			explored.append(node)

	# Condition when the nodes 
	# are not connected
	print("So sorry, but a connecting"\
				"path doesn't exist :(")
	return None


# Code by Eryk Kopczynski
def find_shortest_pathOptimal(graph, start, end):
    dist = {start: [start]}
    
    q = deque(start)
    
    while len(q):
        at = q.popleft()
        for next in graph[at]:
            if next not in dist:
                dist[next] = [dist[at], next]
                q.append(next)
    return dist.get(end)


def bfs(graph, root, compute_distances = False):
    path=[]
    if compute_distances:
        visited, queue = set(), deque([(root, 0)])
    else:
        visited, queue = set(), deque([root])
    visited.add(root)

    steps = 0
    while queue:

        # Dequeue a vertex from queue
        if compute_distances:
            (vertex, steps) = queue.popleft()
        else:
            vertex = queue.popleft() 

        path.append((vertex, steps)) if compute_distances else path.append(vertex)
        #print(str(vertex) + ' ', end='')

        # If not visited, mark it as visited, and
        # enqueue it
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                if compute_distances:
                    queue.append((neighbour, steps + 1))
                else:
                    queue.append(neighbour)

    return path

# DFS algorithm
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)

    #print(start)

    for next in set(graph[start]) - visited:
        dfs(graph, next, visited)
    return visited





# Bellman Ford Algorithm in Python
class Graph2:

    def __init__(self, vertices):
        self.V = vertices   # Total number of vertices in the graph
        self.graph = []     # Array of edges

    # Add edges
    def add_edge(self, s, d, w):
        self.graph.append([s, d, w])

    # Print the solution
    def print_solution(self, dist):
        print("Vertex Distance from Source")
        for i in range(self.V):
            print("{0}\t\t{1}".format(i, dist[i]))

    def bellman_ford(self, src):

        # Step 1: fill the distance array and predecessor array
        dist = [float("Inf")] * self.V
        # Mark the source vertex
        dist[src] = 0

        # Step 2: relax edges |V| - 1 times
        for _ in range(self.V - 1):
            for s, d, w in self.graph:
                if dist[s] != float("Inf") and dist[s] + w < dist[d]:
                    dist[d] = dist[s] + w

        # Step 3: detect negative cycle
        # if value changes then we have a negative cycle in the graph
        # and we cannot find the shortest distances
        for s, d, w in self.graph:
            if dist[s] != float("Inf") and dist[s] + w < dist[d]:
                print("Graph contains negative weight cycle")
                return

        # No negative weight cycle found!
        # Print the distance and predecessor array
        self.print_solution(dist)
    
    # Search function

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    #  Applying Kruskal algorithm
    def kruskal_algo(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)
        for u, v, weight in result:
            print("(%d - %d): %d" % (u, v, weight))

#print(find_all_paths(graphForTests,'A','D'))



class Graph():
    def __init__(self):
        """
        self.edges is a dict of every possible next nodes
        e.g. {'Z': ['A', 'B', 'C',], ...}
        self.weights contains the weights between two nodes,
        ...the two nodes serving as the tuple
        e.g. {('Z', 'A'): 11, ('Z', 'C'): 2.4, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, from_node, to_node, weight):
        # connecting nodes from both sides
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)

        # catering for the source and destination nodes
        self.weights[(from_node, to_node)] = weight
        self.weights[from_node] = weight
        self.weights[(to_node, from_node)] = weight
        # combining the indegree and outdegree weights were possible


    # adapted from https://www.redblobgames.com/pathfinding/a-star/implementation.html
    def a_star_search(self, start, goal):
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = {}
        cost_so_far = {}

        came_from[start] = None
        cost_so_far[start] = 0
        
        while not frontier.empty():
            current: Location = frontier.get()
            
            if current == goal:
                break
            
            for next in self.edges[current]:
                new_cost = cost_so_far[current] + self.weights[(current, next)]

                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost #+ heuristic(next, goal)
                    frontier.put(next, priority)
                    came_from[next] = current
        
        return came_from, cost_so_far



def dijsktra(graph, initial, end):
    # the shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            

            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"

        # the next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # determing the shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path


from collections import defaultdict

class Graph3:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)

    def get_edges(self):
        return self.graph
    
    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def dfs(self, u, visited, discovery_time, low, parent, time, articulation_points):
        visited[u] = True
        discovery_time[u] = time
        low[u] = time
        children = 0

        for v in self.graph[u]:
            if not visited[v]:
                parent[v] = u
                children += 1
                self.dfs(v, visited, discovery_time, low, parent, time + 1, articulation_points)

                low[u] = min(low[u], low[v])

                if parent[u] == 0 and children > 1:
                    articulation_points.add(u)
                elif parent[u] != 0 and low[v] >= discovery_time[u]:
                    articulation_points.add(u)
            elif v != parent[u]:
                low[u] = min(low[u], discovery_time[v])

    def find_articulation_points(self):
        visited = defaultdict(int)
        discovery_time = defaultdict(int)
        low = defaultdict(int)
        parent = defaultdict(int)
        time = 0
        articulation_points = set()

        for u in (self.V):
            print(visited)
            if not visited[u]:
                print(u)
                self.dfs(u, visited, discovery_time, low, parent, time, articulation_points)

        return list(articulation_points)



"""
g = Graph(5)
g.add_edge(0, 1, 5)
g.add_edge(0, 2, 4)
g.add_edge(1, 3, 3)
g.add_edge(2, 1, 6)
g.add_edge(3, 2, 2)

g.bellman_ford(0)

g = Graph(6)
g.add_edge(0, 1, 4)
g.add_edge(0, 2, 4)
g.add_edge(1, 2, 2)
g.add_edge(1, 0, 4)
g.add_edge(2, 0, 4)
g.add_edge(2, 1, 2)
g.add_edge(2, 3, 3)
g.add_edge(2, 5, 2)
g.add_edge(2, 4, 4)
g.add_edge(3, 2, 3)
g.add_edge(3, 4, 3)
g.add_edge(4, 2, 4)
g.add_edge(4, 3, 3)
g.add_edge(5, 2, 2)
g.add_edge(5, 4, 3)
g.kruskal_algo()

#print(find_path(graphForTests,'A','D'))
print(find_all_paths(graphForTests,'A','D'))
#print(find_shortest_path(graphForTests,'A','D'))
#print(find_shortest_pathOptimal(graphForTests,'A','D'))
#print(dfs(gdict,'a'))

print(dfs(gdict,'a'))
print(bfs(gdict,'a'))
printGraph(gdict)
"""

