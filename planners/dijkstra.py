# planners/dijkstra.py
import heapq

def dijkstra_grid(grid, start, goal):
    R,C = grid.shape
    inf = 10**9
    dist = {start: 0}
    parent = {start: None}
    heap = [(0, start)]
    while heap:
        d,u = heapq.heappop(heap)
        if d != dist.get(u, inf):
            continue
        if u == goal:
            path = [goal]
            p = parent[goal]
            while p is not None:
                path.append(p)
                p = parent[p]
            return path[::-1]
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nb = (u[0]+dr, u[1]+dc)
            if not (0 <= nb[0] < R and 0 <= nb[1] < C):
                continue
            if grid[nb] == 1:
                continue
            nd = d + 1
            if nd < dist.get(nb, inf):
                dist[nb] = nd
                parent[nb] = u
                heapq.heappush(heap, (nd, nb))
    return []
