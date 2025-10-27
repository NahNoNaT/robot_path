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
    font = pygame.font.SysFont(None, 24)  # Font for item counts

    for r in range(rows):
        for c in range(cols):
            x = c * (cellsize + MARGIN) + MARGIN
            y = r * (cellsize + MARGIN) + MARGIN
            rect = pygame.Rect(x, y, cellsize, cellsize)

            if gw.grid[r,c] == 1:
                color = COLORS['obstacle']
            elif (r,c) == gw.start:
                color = COLORS['start']
            elif (r,c) in gw.goal_cells:
                if gw.goal_cells[(r,c)] > 0:
                    color = COLORS['goal']
                    # Draw number of remaining items
                    items = str(gw.goal_cells[(r,c)])
                    text = font.render(items, True, COLORS['text'])
                    text_rect = text.get_rect(center=rect.center)
                else:
                    color = COLORS['empty']  # Empty goal cell
            else:
                color = COLORS['empty']
            
            pygame.draw.rect(surface, color, rect)

            # Draw items count for goal cells with remaining items
            if (r,c) in gw.goal_cells and gw.goal_cells[(r,c)] > 0:
                surface.blit(text, text_rect)

def animate_path(gw, path, fps=2, title="Robot", step_delay=5.0, rewards=None, goal_history=None, goal_positions=None):
    """
    rewards: optional list of reward values aligned with transitions (len = len(path)-1)
    goal_history: sequence of goal-count tuples per timestep (aligned with path)
    goal_positions: ordered goal locations matching entries in goal_history tuples
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
    total_reward = 0
    has_goal_history = goal_history is not None and goal_positions is not None
    if has_goal_history and len(goal_history) > 0:
        initial_items = sum(goal_history[0])
    else:
        initial_items = sum(gw.goal_cells.values())
    remaining_items = initial_items  # Track remaining items
    items_collected = 0

    while running:
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(COLORS['bg'])

        if has_goal_history and len(goal_history) > 0:
            state_idx = min(step, len(goal_history) - 1)
            goals_state = goal_history[state_idx]
            for idx, goal_pos in enumerate(goal_positions):
                gw.goal_cells[goal_pos] = goals_state[idx]
                gw.grid[goal_pos] = 2 if goals_state[idx] > 0 else 0

        draw_grid(screen, gw, cellsize)

        # Draw path marker for empty cells only
        for i, p in enumerate(path[:step+1]):
            r, c = p
            if gw.grid[r, c] == 0 and (r, c) not in gw.goal_cells:
                x = c * (cellsize + MARGIN) + MARGIN
                y = r * (cellsize + MARGIN) + MARGIN
                center = (x + cellsize//2, y + cellsize//2)
                pygame.draw.circle(screen, COLORS['path'], center, cellsize//6)

        # Draw robot at current position
        if step < len(path):
            r, c = path[step]
            x = c * (cellsize + MARGIN) + MARGIN
            y = r * (cellsize + MARGIN) + MARGIN
            rect = pygame.Rect(x+4, y+4, cellsize-8, cellsize-8)
            pygame.draw.rect(screen, COLORS['robot'], rect)

        # Update statistics
        current_reward = 0
        if rewards and step-1 >= 0 and step-1 < len(rewards):
            current_reward = rewards[step-1]
            total_reward += current_reward
            
        # Calculate remaining items
        if has_goal_history and len(goal_history) > 0:
            remaining_items = sum(goal_history[min(step, len(goal_history) - 1)])
        else:
            remaining_items = sum(gw.goal_cells.values())
        items_collected = initial_items - remaining_items

        # Update display information
        info_lines = [
            f"Step: {step}/{len(path)-1}",
            f"Current Reward: {current_reward:+.1f}",
            f"Total Reward: {total_reward:+.1f}",
            f"Items Collected: {items_collected}/{initial_items}",
            f"Remaining Items: {remaining_items}"
        ]

        # Draw info text
        for i, line in enumerate(info_lines):
            txt = font.render(line, True, COLORS['text'])
            screen.blit(txt, (10, height-80 + i*18))

        pygame.display.flip()
        step += 1

        # Check if we've reached the end of path
        if step > len(path):
            time.sleep(step_delay)
            running = False

    pygame.quit()
