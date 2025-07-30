import numpy as np
import random
import pygame
from collections import deque, defaultdict
from typing import List, Tuple

CELL_SIZE = 30
PADDING = 2
WALL_COLOR = (47, 79, 79)  # Dark slate gray
PATH_COLOR = (255, 255, 255)  # White
START_COLOR = (50, 205, 50)  # Lime green
END_COLOR = (220, 20, 60)  # Crimson
PLAYER_COLOR = (255, 165, 0)  # Orange
BACKGROUND_COLOR = (169, 169, 169)  # Dark gray
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Keeping the original generate_maze function
def generate_maze(m, n, M, N):
    # Initialize maze with walls
    maze = [['#' for _ in range(N)] for _ in range(M)]
            
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


        i = random.randint(1, m-2)
        j = random.randint(1, n-2)
        maze[i][j] = 'x'
        start = [i, j]
            

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

def resolver_labirinto(start, end, dist, maze_height, maze_width):
    player_pos = tuple(start)
    path = [player_pos]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while player_pos != end:
        y, x = player_pos
        menor = float('inf')
        proximo = None

        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < maze_height and 0 <= nx < maze_width:
                if dist[ny][nx] != -1 and dist[ny][nx] < menor:
                    menor = dist[ny][nx]
                    proximo = (ny, nx)

        path.append(proximo)
        player_pos = proximo

    return path

def resolver_labirinto_semdist(maze, start, maze_height, maze_width):
    player_pos = tuple(start)
    path = [player_pos]
    visited_rewards = defaultdict(lambda: 0)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    max_steps = maze_height * maze_width * 2

    for _ in range(max_steps):
        y, x = player_pos
        best_score = float('-inf')
        next_pos = None

        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < maze_height and 0 <= nx < maze_width:
                if maze[ny][nx] != '#':
                    reward = 0
                    if (ny, nx) not in visited_rewards:
                        reward += 1  #nova célula
                    else:
                        reward -= 1  #já visitada
                        reward += visited_rewards[(ny, nx)] * 0.5  #se já teve valor positivo, pode ser boa

                    if reward > best_score:
                        best_score = reward
                        next_pos = (ny, nx)

        
        path.append(next_pos)
        visited_rewards[next_pos] -= 0.2  

        if maze[next_pos[0]][next_pos[1]] == 'y':
            break  

        player_pos = next_pos

    return path        

class MazeGame:
    def __init__(self, maze_height=10, maze_width=15, max_maze_height=10, max_maze_width=15):
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.max_maze_width = max_maze_width
        self.max_maze_height = max_maze_height
        self.maze = None
        self.player_pos = None
        self.end_pos = None
        self.game_won = False
        self.movimentos = 0
        self.dist = None
        self.rewards = []
        self.reset()

    def reset(self) -> np.ndarray:
        self.maze, self.player_pos, self.end_pos, self.dist = generate_maze(self.maze_height, self.maze_width,self.max_maze_height, self.max_maze_width)
        self.game_won = False
        self.movimentos = 0
        return 
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.movimentos = 0
        previous_x, previous_y = self.player_pos
        new_x, new_y = previous_x, previous_y

        self.prev_positions = deque(maxlen=5)

        reward = 0

        if self.player_pos == self.end_pos:
            reward += 100
            done = True
            self.game_won = True
        elif self.dist[new_x][new_y] == -1: #parede
            reward -= 10
            done = False
        else:   #estímulo pra tentar achar a saída e não ficar parado
            reward -= 1
            done = False

        if self.player_pos in self.prev_positions:
            reward -= 5  # Penaliza repetição
        self.prev_positions.append(self.player_pos)
            
        if self.movimentos > max((self.maze_height * self.maze_width) / 2, 20):
            reward -= 50
            done = True
        
        self.rewards.append(reward)
 
        return self._get_state(), reward, done

class GameVisualizer:
    def __init__(self, maze_height: 10, maze_width: 15, cell_size: int = CELL_SIZE):
        pygame.init()
        self.cell_size = cell_size
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.screen = pygame.display.set_mode((maze_width * cell_size, maze_height * cell_size))
        pygame.display.set_caption("Maze Game - DQN Training")

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
        
        # Draw maze
        for y in range(game.maze_height):
            for x in range(game.maze_width):
                cell = game.maze[y][x]
                if cell == '#':
                    color = WALL_COLOR
                elif (y, x) == game.end_pos:  # Note the (y, x) order
                    color = END_COLOR
                else:
                    color = PATH_COLOR
                self.draw_cell(x, y, color)
        
        # Draw player (convert from [y, x] to [x, y] for display)
        player_y, player_x = game.player_pos  # Unpack in y,x order
        self.draw_cell(player_x, player_y, PLAYER_COLOR)  # Draw in x,y order

        # Draw score and moves
        font = pygame.font.Font(None, 36)
        if score is not None:
            score_text = font.render(f"Score: {score:.1f}", True, WHITE)
            self.screen.blit(score_text, (5, 5))
        
        moves_text = font.render(f"Moves: {game.movimentos}", True, BLACK)
        self.screen.blit(moves_text, (5, 35))

        # Draw win message
        if game.game_won:
            font = pygame.font.Font(None, 74)
            text = font.render('You Won!', True, (0, 255, 0))
            text_rect = text.get_rect(center=(self.maze_width * self.cell_size / 2,
                                            self.maze_height * self.cell_size / 2))
            self.screen.blit(text, text_rect)

        pygame.display.flip()

if __name__ == "__main__":
    visualizer = GameVisualizer(10, 15)  # ou outro tamanho desejado
    clock = pygame.time.Clock()

    running = True
    while running:
        # Cria novo labirinto e resolve
        game = MazeGame(10, 15)
        #solution_path = resolver_labirinto(game.player_pos, game.end_pos, game.dist, game.maze_height, game.maze_width)
        solution_path = resolver_labirinto_semdist(game.maze, game.player_pos, game.maze_height, game.maze_width)

        # Mostra o labirinto sendo resolvido
        index = 0
        while index < len(solution_path):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            game.player_pos = solution_path[index]
            game.movimentos += 1
            index += 1
            if game.player_pos == game.end_pos:
                game.game_won = True

            visualizer.draw_game(game)
            clock.tick(15)  # velocidade da animação

        # Aguarda um pouco antes de reiniciar com novo labirinto
        pygame.time.wait(1000)  # 1 segundo

    pygame.quit()
