# Author: Samuel Lozano
from __future__ import annotations

import numpy as np
import heapq

class AStarPathfinder:
    def __init__(self, grid, wall_value=1):
        self.grid = grid
        self.wall_value = wall_value
        self.n_rows, self.n_cols = grid.shape

    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def neighbors(self, node):
        dirs = [(0,1),(1,0),(0,-1),(-1,0)]
        result = []
        for d in dirs:
            nx, ny = node[0]+d[0], node[1]+d[1]
            if 0 <= nx < self.n_rows and 0 <= ny < self.n_cols:
                if self.grid[nx, ny] != self.wall_value:
                    result.append((nx, ny))
        return result

    def find_path(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for next in self.neighbors(current):
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current
        # reconstruct path
        path = []
        node = goal
        while node and node in came_from:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path if path and path[0] == start else []
