import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
        state.fill(-2)  # Darkness

        # Show area around player
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
        previous_x, previous_y = self.player_pos
        new_x, new_y = previous_x, previous_y
        
        if action == 0:  # Left
            new_x = max(0, previous_x - 1)
        elif action == 1:  # Up
            new_y = max(0, previous_y - 1)
        elif action == 2:  # Right
            new_x = min(self.maze_width - 1, previous_x + 1)
        elif action == 3:  # Down
            new_y = min(self.maze_height - 1, previous_y + 1)

        if 0 <= new_x < self.maze_height and 0 <= new_y < self.maze_width:
            if self.maze[new_x][new_y] != "#":
                self.player_pos = (new_x, new_y)

        reward = 0

        if self.player_pos == self.end_pos:
            reward += 1000
            done = True
            self.game_won = True
        elif self.dist[new_x][new_y] == -1:
            reward -= 20
            done = False
        elif self.dist[new_x][new_y] > self.dist[previous_x][previous_y]:
            reward -= 15
            done = False
        elif self.dist[new_x][new_y] < self.dist[previous_x][previous_y]:
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

class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.dones[indices])
        )

    def __len__(self) -> int:
        return self.size

class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural Network
        self.policy_net = DQN(state_size, 256, action_size).to(self.device)
        self.target_net = DQN(state_size, 256, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize with these values
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001) #included a learning rate to make it more stable
        self.memory = ReplayMemory(20000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999 #more exploration
        self.target_update = 5
        
    def save(self, path="maze_agent.pth"):
        """Saves the agent's models and training state"""
        save_dict = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(save_dict, path)
        print(f"Agent saved to {path}")
        
    def load(self, path="maze_agent.pth"):
        """Loads the agent's models and training state"""
        if os.path.exists(path):
            # Use weights_only=True for better security
            checkpoint = torch.load(path, weights_only=True)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            print(f"Agent loaded from {path}")
            return True
        else:
            print(f"No saved model found at {path}")
            return False

    def get_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Get samples as tensors directly from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class LogLabirinto:
    def __init__(self, maze, player_pos, end_pos, test_number=1, log_file="maze_tests.log"):
        self.initial_maze = [row[:] for row in maze]
        self.player_pos = player_pos  # Start position
        self.end_pos = end_pos
        self.test_number = test_number
        self.movements = []
        self.log_file = log_file  # Arquivo único para todos os testes

        #Adicionar cabeçalho do novo teste com data e hora
        with open(self.log_file, "a") as f:  #Modo append para não sobrescrever
            f.write("\n" + "="*50 + "\n")  #Separador entre testes
            f.write(f"Teste #{test_number} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Labirinto Inicial:\n")
            self._write_maze_to_file(f)

    def _write_maze_to_file(self, file):
        """Escreve o labirinto inicial no arquivo de log."""
        for i in range(len(self.initial_maze)):
            row = []
            for j in range(len(self.initial_maze[0])):
                if (i, j) == self.player_pos:
                    row.append("P")  # Player
                elif (i, j) == self.end_pos:
                    row.append("E")  # End position
                else:
                    row.append(self.initial_maze[i][j])
            file.write("".join(row) + "\n")
        file.write("\n")  # Linha em branco para separar

    def log_action(self, action):
        """Registra cada movimento no log."""
        action_names = {0: "Left", 1: "Up", 2: "Right", 3: "Down"}
        action_name = action_names.get(action, f"Unknown({action})")
        self.movements.append(action_name)

        with open(self.log_file, "a") as f:
            f.write(f"Movimento #{len(self.movements)}: {action_name}\n")

    def save_summary(self, success=None):
        """Salva o resumo final do teste no log."""
        with open(self.log_file, "a") as f:
            f.write("\n Resumo do Teste:\n")
            for i, move in enumerate(self.movements):
                f.write(f"{i+1}. {move}\n")

            f.write(f"\n Total de movimentos: {len(self.movements)}\n")
            if success is not None:
                f.write(f"Labirinto resolvido: {'Sim' if success else 'Não'}\n")
            f.write("="*50 + "\n")

def train_agent(episodes: int = 1000, render: bool = True, render_delay: float = 0.1,
                save_interval: int = 100, save_path: str = "maze_agent.pth",
                load_model: bool = False) -> Tuple[DQNAgent, List[float]]:
    env = MazeGame()
    state_size = env.maze_width * env.maze_height
    agent = DQNAgent(state_size, 4)  # 4 actions: up, down, left, right
    scores = []
    
    # Try to load existing model if requested
    if load_model:
        if agent.load(save_path):
            print("Continuing training from saved model")
        else:
            print("Starting fresh training")
    
    if render:
        visualizer = GameVisualizer(env.maze_width, env.maze_height)

    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                # Handle Pygame events
                if render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt

                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.memory.push(state, action, reward, next_state, done)
                agent.train()
                total_reward += reward
                state = next_state

                if render:
                    visualizer.draw_game(env, total_reward)
                    time.sleep(render_delay)

            scores.append(total_reward)

            if episode % agent.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                
            # Save at regular intervals
            if (episode + 1) % save_interval == 0:
                agent.save(save_path)

            if (episode + 1) % 100 == 0:
                avg_score = np.mean(scores[-100:])
                print(f"Episode: {episode+1}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save on interrupt
        agent.save(save_path)

    finally:
        if render:
            pygame.quit()

    return agent, scores

def demonstrate_agent(agent: DQNAgent, num_games: int = 10, render_delay: float = 0.2, log=None):
    """Demonstrates the trained agent's performance"""
    env = MazeGame()
    visualizer = GameVisualizer(env.maze_width, env.maze_height)

    original_epsilon = agent.epsilon
    agent.epsilon = 0

    try:
        total_score = 0
        for game in range(num_games):
            state = env.reset()
            
            #Create a new log for each game if log is provided
            if log is not None:
                game_log = LogLabirinto(env.maze, env.player_pos, env.end_pos, 
                                      test_number=game+1, 
                                      log_file=f"agent_game_{game+1}.txt")
            
            game_score = 0
            done = False

            while not done:
                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                # Get action from trained agent
                action = agent.get_action(state)
                
                # Log the action if logging is enabled
                if log is not None:
                    game_log.log_action(action)
                    
                next_state, reward, done = env.step(action)
                game_score += reward
                state = next_state

                # Update visualization
                visualizer.draw_game(env, game_score)
                time.sleep(render_delay)

            # Save summary of the game
            if log is not None:
                game_log.save_summary(success=env.game_won)
                
            total_score += game_score
            print(f"Game {game + 1}: Score = {game_score:.2f}")

        print(f"\nAverage Score over {num_games} games: {total_score/num_games:.2f}")

    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")

    finally:
        pygame.quit()
        # Restore original epsilon
        agent.epsilon = original_epsilon

if __name__ == "__main__":
    # First train the agent
    env = MazeGame()
    state_size = env.maze_width * env.maze_height
    
    print("Training agent...")
    agent, scores = train_agent(
        episodes=500, 
        render=True, 
        render_delay=0.01,
        save_interval=100,  # Save every 100 episodes
        load_model=True    # Try to load existing model
    )
    print(f"Final 100 episodes average score: {np.mean(scores[-100:]):.2f}")
    
    # Save the final model
    agent.save("maze_agent_final.pth")

    # Then demonstrate its performance with logging
    print("\nNow watching the trained agent play...")
    
    # Create a log instance for demonstration runs
    log = LogLabirinto(env.maze, env.player_pos, env.end_pos, 
                     test_number=1, log_file="agent_demonstration.txt")
    
    # Pass the log to a modified demonstrate_agent function
    demonstrate_agent(agent, num_games=20, render_delay=0.2, log=log)