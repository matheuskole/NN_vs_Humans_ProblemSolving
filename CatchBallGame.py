import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pygame
import time
from collections import deque
from typing import List, Tuple

class CatchGame:
    #inicializa o jogo com um espaço de 10x10
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.reset()

    #não entendi muito bem esse reset
    #o player começa no meio, a bola é gerada aleatoriamente no topo
    def reset(self) -> np.ndarray:
        """Reset game state and return initial observation"""
        self.player_x = self.width // 2
        self.ball_x = random.randint(0, self.width-1)
        self.ball_y = 0
        return self._get_state()

    #cria uma matriz com 0's, onde 1 representa a posição da bola e 2 o player
    #utiliza a função flatten para deixar a matriz em forma de vetor, necessário para a rede neural
    def _get_state(self) -> np.ndarray:
        """Convert game state to neural network input"""
        state = np.zeros((self.height, self.width))
        state[self.ball_y, self.ball_x] = 1  # Ball position
        state[self.height-1, self.player_x] = 2  # Player position
        return state.flatten()

    #define como o player pode se movimentar ou escolher ficar parado, steps
    #enquanto isso, a bola vai caindo += 1
    #se a bola estiver na altura do chão, acabou
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Take action and return new_state, reward, done"""
        # Action: 0 = left, 1 = stay, 2 = right
        previous_x = self.player_x
        if action == 0:
            self.player_x = max(0, self.player_x - 1)
        elif action == 2:
            self.player_x = min(self.width - 1, self.player_x + 1)

        self.ball_y += 1

        done = self.ball_y == self.height - 1

        #reinforcement learning, pois da uma recompensa caso o player tenha pegado a bola e retira caso não pegue
        #quanto maior for a diferença, mais punido o player será
        reward = 0
        if done:
            if abs(self.player_x - self.ball_x) <= 1:  #Obteve sucesso
                reward = 100
            else:
                reward = -100
        else:
             if self.player_x == self.ball_x:
                 reward = 10
             elif abs(self.player_x - self.ball_x) < abs(previous_x - self.ball_x) :
                reward = 10
             else:
                reward = -10

        return self._get_state(), reward, done

class GameVisualizer:
    def __init__(self, width: int, height: int, cell_size: int = 50):
        pygame.init()
        self.cell_size = cell_size
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption("Catch Game - DQN Training")

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)

    def draw_game(self, game: CatchGame, reward: float = None):
        self.screen.fill(self.BLACK)

        # Draw grid
        for x in range(self.width):
            for y in range(self.height):
                pygame.draw.rect(self.screen, self.WHITE,
                               (x * self.cell_size, y * self.cell_size,
                                self.cell_size, self.cell_size), 1)

        # Draw ball
        pygame.draw.circle(self.screen, self.RED,
                         (game.ball_x * self.cell_size + self.cell_size//2,
                          game.ball_y * self.cell_size + self.cell_size//2),
                         self.cell_size//3)

        # Draw player
        pygame.draw.rect(self.screen, self.BLUE,
                        (game.player_x * self.cell_size,
                         (self.height-1) * self.cell_size,
                         self.cell_size, self.cell_size))

        # Draw reward if provided
        if reward is not None:
            font = pygame.font.Font(None, 36)
            reward_text = font.render(f"Reward: {reward}", True, self.WHITE)
            self.screen.blit(reward_text, (10, 10))

        pygame.display.flip()

#rede neural é construída com 4 camadas, input, hidden, output
#nn.Module configura o pytorch para que tenha acesso a todas as funcionalidades
#nn.Sequential empilha a rede, a saída de um, é entrada do outro
#o output size nesse caso é o número de ações que o player pode tomar
#DQN é uma rede de aprendizado por reforço, treina para aprimorar a rede
class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    #feedforward/fluxo de dados
    def forward(self, x):
        return self.network(x)

    #parte do DQN, tem estado: situação atual,ação: ação do player a partir do estado, recompensa: a recompensa que teve, prox_estado: estado pós ação
    #se aprimorarmos mais, poderiamos atualizar a recompensa a cada ação do player, levaria menos tentativas
class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List:
        return random.sample(self.memory, batch_size)   #batch_size é a quantidade de testes para uma atualização dos valores da RN

    def __len__(self) -> int:
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Rede Neural
        self.policy_net = DQN(state_size, 128, action_size).to(self.device)
        self.target_net = DQN(state_size, 128, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        #Inicializa com esses valores
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10

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

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_agent(episodes: int = 1000, render: bool = True, render_delay: float = 0.1) -> List[float]:
    env = CatchGame()
    state_size = env.width * env.height
    agent = DQNAgent(state_size, 3)
    scores = []

    if render:
        visualizer = GameVisualizer(env.width, env.height)

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

            if (episode + 1) % 100 == 0:
                avg_score = np.mean(scores[-100:])
                print(f"Episode: {episode+1}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        if render:
            pygame.quit()

    return agent, scores

def demonstrate_agent(agent: DQNAgent, num_games: int = 10, render_delay: float = 0.2):
    """
    Demonstrates the trained agent's performance
    """
    env = CatchGame()
    visualizer = GameVisualizer(env.width, env.height)

    # Disable exploration for demonstration
    original_epsilon = agent.epsilon
    agent.epsilon = 0

    try:
        total_score = 0
        for game in range(num_games):
            state = env.reset()
            game_score = 0
            done = False

            while not done:
                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                # Get action from trained agent
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                game_score += reward
                state = next_state

                # Update visualization
                visualizer.draw_game(env, game_score)
                time.sleep(render_delay)

            total_score += game_score
            print(f"Game {game + 1}: Score = {game_score}")

        print(f"\nAverage Score over {num_games} games: {total_score/num_games:.2f}")

    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")

    finally:
        pygame.quit()
        # Restore original epsilon
        agent.epsilon = original_epsilon

if __name__ == "__main__":
    # First train the agent
    env = CatchGame()
    state_size = env.width * env.height
    #agent = DQNAgent(state_size, 3)

    print("Training agent...")
    agent,scores = train_agent(1000, render=True, render_delay=0.01)  # Fast training visualization
    print(f"Final 1000 episodes average score: {np.mean(scores[-100:]):.2f}")

    # Then demonstrate its performance
    print("\nNow watching the trained agent play...")
    demonstrate_agent(agent, num_games=10, render_delay=0.2)  # Slower visualization for demonstration