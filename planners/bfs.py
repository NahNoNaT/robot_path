# planners/bfs.py
from collections import deque

def bfs_grid(grid, start, goal):
    """
    grid: 2D numpy array (0 empty, 1 obstacle)
    start, goal: (r,c)
    returns: path list of positions from start to goal (inclusive) or [] if not found.
    """
    if start == goal:
        return [start]
    R,C = grid.shape
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nb = (cur[0]+dr, cur[1]+dc)
            if not (0 <= nb[0] < R and 0 <= nb[1] < C):
                continue
            if grid[nb] == 1:
                continue
            if nb in parent:
                continue
            parent[nb] = cur
            if nb == goal:
                # reconstruct
                path = [goal]
                p = cur
                while p is not None:
                    path.append(p)
                    p = parent[p]
                return path[::-1]
            q.append(nb)
    return []
