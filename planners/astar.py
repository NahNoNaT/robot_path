# planners/astar.py
import heapq
from utils import manhattan

def astar_grid(grid, start, goal):
    R,C = grid.shape
    def h(a,b):
        return manhattan(a,b)
    open_heap = [(h(start,goal), 0, start)]
    parent = {start: None}
    gscore = {start: 0}
    closed = set()
    while open_heap:
        f, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [goal]
            p = parent[goal]
            while p is not None:
                path.append(p)
                p = parent[p]
            return path[::-1]
        closed.add(cur)
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nb = (cur[0]+dr, cur[1]+dc)
            if not (0 <= nb[0] < R and 0 <= nb[1] < C):
                continue
            if grid[nb] == 1:
                continue
            tentative_g = g + 1
            if tentative_g < gscore.get(nb, float('inf')):
                gscore[nb] = tentative_g
                parent[nb] = cur
                heapq.heappush(open_heap, (tentative_g + h(nb, goal), tentative_g, nb))
    return []
