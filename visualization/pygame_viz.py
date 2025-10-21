"""
Simple Pygame visualization for grid and a given path (list of positions).
Call draw_run(gridworld, paths_list, start) where paths_list is list of waypoints to animate.
"""
import pygame
import sys
import time

CELL = 40
MARGIN = 2
COLORS = {
    'bg': (30,30,30),
    'empty': (220,220,220),
    'obstacle': (40,40,40),
    'goal': (191, 143, 0),
    'start': (30,144,255),
    'robot': (255,50,50),
    'path': (100,255,100),
    'text': (240,240,240)
}

def draw_grid(surface, gw, cellsize):
    rows, cols = gw.grid.shape
    for r in range(rows):
        for c in range(cols):
            x = c * (cellsize + MARGIN) + MARGIN
            y = r * (cellsize + MARGIN) + MARGIN
            rect = pygame.Rect(x, y, cellsize, cellsize)
            if gw.grid[r,c] == 1:
                color = COLORS['obstacle']
            elif (r,c) == gw.start:
                color = COLORS['start']
            elif (r,c) in gw.goal_cells and gw.goal_cells[(r,c)]>0:
                color = COLORS['goal']
            else:
                color = COLORS['empty']
            pygame.draw.rect(surface, color, rect)

def animate_path(gw, path, fps=2, title="Robot", step_delay=2.0, rewards=None):
    """
    rewards: optional list of reward values aligned with transitions (len = len(path)-1)
    """
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont(None, 20)
    rows, cols = gw.grid.shape
    cellsize = CELL
    width = cols * (cellsize + MARGIN) + MARGIN
    height = rows * (cellsize + MARGIN) + MARGIN + 60
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    running = True
    step = 0
    while running:
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(COLORS['bg'])
        draw_grid(screen, gw, cellsize)
        # draw path up to current step
        for i,p in enumerate(path[:step+1]):
            r,c = p
            x = c * (cellsize + MARGIN) + MARGIN
            y = r * (cellsize + MARGIN) + MARGIN
            center = (x + cellsize//2, y + cellsize//2)
            pygame.draw.circle(screen, COLORS['path'], center, cellsize//6)
        # robot at current
        if step < len(path):
            r,c = path[step]
            x = c * (cellsize + MARGIN) + MARGIN
            y = r * (cellsize + MARGIN) + MARGIN
            rect = pygame.Rect(x+4, y+4, cellsize-8, cellsize-8)
            pygame.draw.rect(screen, COLORS['robot'], rect)
        # draw text: current step, reward (if available) and items remaining at current cell
        cur_pos = path[min(step, len(path)-1)]
        reward_text = ""
        if rewards and step-1 >= 0 and step-1 < len(rewards):
            reward_text = f"Reward (last step): {rewards[step-1]}"
        else:
            # simple on-the-fly reward heuristic: 1 if this cell had items >0
            if cur_pos in gw.goal_cells and gw.goal_cells[cur_pos] > 0:
                reward_text = "Reward (heuristic): 1"
            else:
                reward_text = "Reward (heuristic): 0"
        items_cnt = gw.goal_cells.get(cur_pos, 0)
        info_lines = [
            f"Step: {step}/{len(path)-1}",
            reward_text,
            f"Items at current cell: {items_cnt}"
        ]
        for i, line in enumerate(info_lines):
            txt = font.render(line, True, COLORS['text'])
            screen.blit(txt, (10, height-50 + i*18))
        pygame.display.flip()
        step += 1
        if step > len(path):
            time.sleep(step_delay)
            running = False
    pygame.quit()
