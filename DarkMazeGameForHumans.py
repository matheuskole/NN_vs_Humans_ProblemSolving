import numpy as np
import random
import pygame
import time
import os
from collections import deque
from typing import List, Tuple
import os
from datetime import datetime

CELL_SIZE = 30
PADDING = 2
WALL_COLOR = (47, 79, 79)  # Dark slate gray
PATH_COLOR = (10, 10, 10)  # Almost black for unlit areas
VISIBLE_PATH_COLOR = (200, 200, 200)  # Light gray for lit paths
START_COLOR = (50, 205, 50)  # Lime green
END_COLOR = (220, 20, 60)  # Crimson
PLAYER_COLOR = (0, 255, 0)  # Green for player
BACKGROUND_COLOR = (0, 0, 0)  # Pure black
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Keeping the original generate_maze function
def generate_maze(m, n):
    # Initialize maze with walls
    maze = [['#' for _ in range(n)] for _ in range(m)]
            
    # Generate random walls
    walls = int((m-2) * (n-2) * 0.3)  # 30% of internal cells will be walls
    done = False
    while not done:
        # Create empty cells
        for i in range(1, m-1):
            for j in range(1, n-1):
                maze[i][j] = '0'
            
        for _ in range(walls):
            i = random.randint(1, m-2)
            j = random.randint(1, n-2)
            maze[i][j] = '#'

        while True:
            i = random.randint(1, m-2)
            j = random.randint(1, n-2)
            if maze[i][j] == '0':
                maze[i][j] = 'x'
                start = [i, j]
                break

        for t in range(1,10):
            i = random.randint(1, m-2)
            j = random.randint(1, n-2)
            if maze[i][j] == '0':
                temp_maze = [row[:] for row in maze]
                if has_path(temp_maze, tuple(start), (i, j))[0]:
                    maze[i][j] = 'y'
                    end = (i, j)
                    done = True
                    break

    dist = [[-1 for _ in range(n)] for _ in range(m)]
    for i in range(0,m):
        for j in range(0,n):
            if maze[i][j] != "#":
                a = has_path(maze, (i,j), end)
                dist[i][j] = a[1]

    return maze, start, end, dist

def has_path(maze, start, end):
    m, n = len(maze), len(maze[0])
    visited = set()
    dist = {}
    dist[start] = 0
    queue = deque([start])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        i, j = queue.popleft()
        if (i, j) == end:
            return True, dist[(i,j)]
            
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if (0 <= ni < m and 0 <= nj < n and 
                maze[ni][nj] != '#' and 
                (ni, nj) not in visited):
                visited.add((ni, nj))
                dist[(ni,nj)] = dist[(i,j)]+1
                queue.append((ni, nj))
   
    return False, -1

class MazeGame:
    def __init__(self, maze_width=15, maze_height=10):
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.maze = None
        self.player_pos = None
        self.end_pos = None
        self.game_won = False
        self.movimentos = 0
        self.dist = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.maze, self.player_pos, self.end_pos, self.dist = generate_maze(self.maze_height, self.maze_width)
        self.game_won = False
        self.movimentos = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        state = np.zeros((self.maze_height, self.maze_width))
        state.fill(-2) 

        player_y, player_x = self.player_pos
        for y in range(max(0, player_y - 1), min(self.maze_height, player_y + 2)):
            for x in range(max(0, player_x - 1), min(self.maze_width, player_x + 2)):
                if self.maze[y][x] == '#':
                    state[y, x] = -1  
                elif (y, x) == self.end_pos:
                    state[y, x] = 2  
                else:
                    state[y, x] = 0  

        state[player_y, player_x] = 1
        
        return state.flatten()

    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.movimentos += 1
        previous_y, previous_x = self.player_pos
        new_y, new_x = previous_y, previous_x
        
        if action == 0:  # Left
            new_x = max(0, previous_x - 1)
        elif action == 1:  # Up
            new_y = max(0, previous_y - 1)
        elif action == 2:  # Right
            new_x = min(self.maze_width - 1, previous_x + 1)
        elif action == 3:  # Down
            new_y = min(self.maze_height - 1, previous_y + 1)

        if 0 <= new_y < self.maze_height and 0 <= new_x < self.maze_width:
            if self.maze[new_y][new_x] != "#":
                self.player_pos = (new_y, new_x)

        reward = 0

        if self.player_pos == self.end_pos:
            reward += 1000
            done = True
            self.game_won = True
        elif self.dist[new_y][new_x] == -1:
            reward -= 20
            done = False
        elif self.dist[new_y][new_x] > self.dist[previous_y][previous_x]:
            reward -= 15
            done = False
        elif self.dist[new_y][new_x] < self.dist[previous_y][previous_x]:
            reward += 15
            done = False
        else:
            reward += 1
            done = False
            
        if self.movimentos > (self.maze_height * self.maze_width) / 2:
            reward -= 50
            done = True

        return self._get_state(), reward, done

class GameVisualizer:
    def __init__(self, maze_width: int, maze_height: int, cell_size: int = CELL_SIZE):
        pygame.init()
        self.cell_size = cell_size
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.screen = pygame.display.set_mode((maze_width * cell_size, maze_height * cell_size))
        pygame.display.set_caption("Dark Maze Game - Square Visibility")

    def run(self, game: MazeGame, player_id=1):
        running = True
        total_score = 0

        game_log = LogLabirinto(game.maze, game.player_pos, game.end_pos, 
                              player_id=player_id, log_file="Dados_Humanos.txt")

        self.draw_game(game, total_score)
        pygame.display.flip()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    game_log.save_summary(success=game.game_won, total_score=total_score)
                    break
                    
                elif event.type == pygame.KEYDOWN:
                    action = None
                    if event.key == pygame.K_LEFT:
                        action = 0
                    elif event.key == pygame.K_UP:
                        action = 1
                    elif event.key == pygame.K_RIGHT:
                        action = 2
                    elif event.key == pygame.K_DOWN:
                        action = 3
                    else:
                        continue
                    
                    if action is not None:
                        _, reward, done = game.step(action)

                        game_log.log_action(action, reward)
                        
                        total_score += reward
                        
                        self.draw_game(game, total_score)
                        pygame.display.flip()
                        
                        if done or game.game_won:
                            game_log.save_summary(success=game.game_won, total_score=total_score)
                            
                            #Resetar para o próximo jogo
                            game.reset()
                            total_score = 0
                            
                            #Cria novo log pro próximo jogo, mesmo ID
                            game_log = LogLabirinto(game.maze, game.player_pos, game.end_pos, 
                                                  player_id=player_id, log_file="Dados_Humanos.txt")
                            
                            #Draw the new game state
                            self.draw_game(game, total_score)
                            pygame.display.flip()  

    def draw_cell(self, x: int, y: int, color: Tuple[int, int, int]):
        rect = pygame.Rect(
            x * self.cell_size + PADDING,
            y * self.cell_size + PADDING,
            self.cell_size - 2 * PADDING,
            self.cell_size - 2 * PADDING
        )
        pygame.draw.rect(self.screen, color, rect)

    def draw_game(self, game: MazeGame, score: float = None):
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw grid lines
        for i in range(game.maze_height + 1):
            pygame.draw.line(self.screen, (50, 50, 50), 
                           (0, i * self.cell_size), 
                           (game.maze_width * self.cell_size, i * self.cell_size))
        for j in range(game.maze_width + 1):
            pygame.draw.line(self.screen, (50, 50, 50), 
                           (j * self.cell_size, 0), 
                           (j * self.cell_size, game.maze_height * self.cell_size))
        
        # Draw visible cells
        player_y, player_x = game.player_pos
        for y in range(max(0, player_y - 1), min(game.maze_height, player_y + 2)):
            for x in range(max(0, player_x - 1), min(game.maze_width, player_x + 2)):
                if (y, x) != game.player_pos:  # Don't draw over player
                    cell = game.maze[y][x]
                    if cell == '#':
                        color = WALL_COLOR
                    elif (y, x) == game.end_pos:
                        color = END_COLOR
                    else:
                        color = VISIBLE_PATH_COLOR
                    self.draw_cell(x, y, color)
        
        # Draw player
        self.draw_cell(player_x, player_y, PLAYER_COLOR)

        font = pygame.font.Font(None, 36)
        if score is not None:
            score_text = font.render(f"Score: {score:.1f}", True, WHITE)
            self.screen.blit(score_text, (5, 5))
        
        moves_text = font.render(f"Moves: {game.movimentos}", True, WHITE)
        self.screen.blit(moves_text, (5, 35))

        if game.game_won:
            font = pygame.font.Font(None, 74)
            text = font.render('You Won!', True, (0, 255, 0))
            text_rect = text.get_rect(center=(self.maze_width * self.cell_size / 2,
                                            self.maze_height * self.cell_size / 2))
            self.screen.blit(text, text_rect)

        pygame.display.flip()

class LogLabirinto:
    def __init__(self, maze, player_pos, end_pos, player_id=1, log_file="Dados_Humanos.txt"):
        self.initial_maze = maze  
        self.player_pos = player_pos
        self.end_pos = end_pos
        self.player_id = player_id
        self.movements = []
        self.rewards = []  
        self.log_file = log_file

        with open(self.log_file, "a") as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"Player #{player_id} - Game - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Initial Maze:\n")
            self._write_maze_to_file(f)

    def _write_maze_to_file(self, file):
        """Desenha o Labirinto Inicial"""
        for i in range(len(self.initial_maze)):
            row = []
            for j in range(len(self.initial_maze[0])):
                if (i, j) == self.player_pos:
                    row.append("S")  # Player
                elif (i, j) == self.end_pos:
                    row.append("E")  # End position
                else:
                    row.append(self.initial_maze[i][j])
            file.write("".join(row) + "\n")
        file.write("\n")  #Linha para separar

    def log_action(self, action, reward=0):
        """Guarda cada movimento"""
        action_names = {0: "Left", 1: "Up", 2: "Right", 3: "Down"}
        action_name = action_names.get(action, f"Unknown({action})")
        self.movements.append(action_name)
        self.rewards.append(reward)

        with open(self.log_file, "a") as f:
            f.write(f"Movement #{len(self.movements)}: {action_name}, Reward: {reward}\n")

    def save_summary(self, success=None, total_score=0):
        """Resumo das Atividades"""
        with open(self.log_file, "a") as f:
            f.write("\nGame Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total movements: {len(self.movements)}\n")
            f.write(f"Total score: {total_score}\n")
            if success is not None:
                f.write(f"Maze solved: {'Yes' if success else 'No'}\n")
            f.write("="*50 + "\n")

if __name__ == "__main__":
    try:
        player_id = int(input("Enter player ID (1, 2, 3, etc.): "))
    except ValueError:
        player_id = 1  
    
    game = MazeGame(maze_width=15, maze_height=10)
    visualizer = GameVisualizer(maze_width=15, maze_height=10)
    visualizer.run(game, player_id)
    
    pygame.quit()    