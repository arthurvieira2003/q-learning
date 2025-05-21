import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

class GridWorld:
    def __init__(self, height=5, width=5, start_pos=(0, 0), goal_pos=(4, 4), obstacles=None):
        self.height = height
        self.width = width
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = [] if obstacles is None else obstacles
        self.agent_pos = start_pos
        
        # Ações possíveis: 0: cima, 1: direita, 2: baixo, 3: esquerda
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ["Cima", "Direita", "Baixo", "Esquerda"]
    
    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos
    
    def step(self, action):
        # Aplicar a ação escolhida
        move = self.actions[action]
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])
        
        # Verificar se o movimento é válido (dentro dos limites e não colide com obstáculos)
        if (0 <= new_pos[0] < self.height and 
            0 <= new_pos[1] < self.width and 
            new_pos not in self.obstacles):
            self.agent_pos = new_pos
        
        # Atribuir recompensa
        if self.agent_pos == self.goal_pos:
            reward = 1.0  # Recompensa por alcançar o objetivo
            done = True
        else:
            reward = -0.01  # Pequena penalidade por cada passo para incentivar caminhos mais curtos
            done = False
            
        return self.agent_pos, reward, done

    def get_state_index(self, pos):
        return pos[0] * self.width + pos[1]
    
    def get_pos_from_index(self, index):
        return (index // self.width, index % self.width)
    
    def render(self, q_table=None, episode=None, show_values=False, show_policy=False):
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 2, width_ratios=[2, 1])
        
        # Grid World
        ax1 = fig.add_subplot(gs[0])
        ax1.set_xlim(0, self.width)
        ax1.set_ylim(0, self.height)
        ax1.set_xticks(np.arange(0, self.width + 1, 1))
        ax1.set_yticks(np.arange(0, self.height + 1, 1))
        ax1.grid(True)
        ax1.set_aspect('equal')
        
        # Inverter o eixo y para que (0,0) fique no canto superior esquerdo
        ax1.set_ylim(self.height, 0)
        
        # Desenhar o ambiente
        for i in range(self.height):
            for j in range(self.width):
                # Adicionar texto para valores Q ou política
                if q_table is not None and (show_values or show_policy):
                    state_idx = self.get_state_index((i, j))
                    
                    if (i, j) in self.obstacles:
                        rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='black')
                        ax1.add_patch(rect)
                    elif (i, j) == self.goal_pos:
                        rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='green', alpha=0.3)
                        ax1.add_patch(rect)
                        if show_values:
                            ax1.text(j + 0.5, i + 0.5, "Meta", ha='center', va='center', fontsize=10)
                    else:
                        # Mostrar os valores Q ou a política
                        if show_values:
                            best_action = np.argmax(q_table[state_idx])
                            value = q_table[state_idx, best_action]
                            ax1.text(j + 0.5, i + 0.5, f"{value:.2f}", ha='center', va='center', fontsize=9)
                        
                        if show_policy:
                            best_action = np.argmax(q_table[state_idx])
                            action_text = ["↑", "→", "↓", "←"][best_action]
                            ax1.text(j + 0.5, i + 0.25, action_text, ha='center', va='center', fontsize=14)
        
        # Destacar a posição atual do agente
        agent_j, agent_i = self.agent_pos[1], self.agent_pos[0]
        agent_circle = plt.Circle((agent_j + 0.5, agent_i + 0.5), 0.3, color='blue')
        ax1.add_patch(agent_circle)
        
        # Destacar a posição inicial
        start_j, start_i = self.start_pos[1], self.start_pos[0]
        start_rect = patches.Rectangle((start_j, start_i), 1, 1, linewidth=1, edgecolor='black', facecolor='yellow', alpha=0.3)
        ax1.add_patch(start_rect)
        
        # Destacar a posição do objetivo
        goal_j, goal_i = self.goal_pos[1], self.goal_pos[0]
        goal_rect = patches.Rectangle((goal_j, goal_i), 1, 1, linewidth=1, edgecolor='black', facecolor='green', alpha=0.3)
        ax1.add_patch(goal_rect)
        
        # Destacar obstáculos
        for obstacle in self.obstacles:
            obs_i, obs_j = obstacle
            obs_rect = patches.Rectangle((obs_j, obs_i), 1, 1, linewidth=1, edgecolor='black', facecolor='black')
            ax1.add_patch(obs_rect)
        
        if episode is not None:
            ax1.set_title(f'Grid World - Episódio {episode}')
        else:
            ax1.set_title('Grid World')
        
        # Informações gerais e legenda
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        info_text = (
            "Ambiente Grid World\n\n"
            f"Dimensões: {self.height}x{self.width}\n"
            f"Posição inicial: {self.start_pos}\n"
            f"Objetivo: {self.goal_pos}\n"
            f"Obstáculos: {len(self.obstacles)}\n\n"
            "Legenda:\n"
            "🟦 - Agente\n"
            "🟨 - Posição inicial\n"
            "🟩 - Objetivo\n"
            "⬛ - Obstáculo\n\n"
        )
        
        if q_table is not None:
            info_text += (
                "Ações:\n"
                "↑ - Cima\n"
                "→ - Direita\n"
                "↓ - Baixo\n"
                "← - Esquerda\n"
            )
        
        ax2.text(0, 0.95, info_text, va='top', fontsize=10)
        
        plt.tight_layout()
        plt.show()

def q_learning(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.995):
    """
    Implementação do algoritmo Q-Learning.
    
    Args:
        env: ambiente GridWorld
        num_episodes: número de episódios para treinamento
        alpha: taxa de aprendizado
        gamma: fator de desconto
        epsilon: parâmetro de exploração inicial
        min_epsilon: valor mínimo de epsilon
        epsilon_decay: taxa de decaimento de epsilon
    
    Returns:
        Q-table treinada
    """
    # Inicializar Q-table
    num_states = env.height * env.width
    num_actions = len(env.actions)
    q_table = np.zeros((num_states, num_actions))
    
    # Métricas para análise
    episode_rewards = []
    episode_steps = []
    
    for episode in range(num_episodes):
        state = env.reset()
        state_idx = env.get_state_index(state)
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Política epsilon-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(num_actions)  # Exploração
            else:
                action = np.argmax(q_table[state_idx])  # Exploração
            
            # Executar ação
            next_state, reward, done = env.step(action)
            next_state_idx = env.get_state_index(next_state)
            
            # Atualizar Q-value usando a equação de Bellman com Diferença Temporal
            old_value = q_table[state_idx, action]
            next_max = np.max(q_table[next_state_idx])
            
            # Equação Q-Learning: Q(s,a) = Q(s,a) + α * [R + γ * max(Q(s',a')) - Q(s,a)]
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state_idx, action] = new_value
            
            # Atualizar estado
            state = next_state
            state_idx = next_state_idx
            
            total_reward += reward
            steps += 1
        
        # Registrar métricas
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Reduzir epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Mostrar progresso a cada 100 episódios
        if (episode + 1) % 100 == 0 or episode == 0:
            print(f"Episódio {episode + 1}/{num_episodes}, Recompensa: {total_reward:.2f}, Passos: {steps}, Epsilon: {epsilon:.4f}")
    
    return q_table, episode_rewards, episode_steps

def test_policy(env, q_table, render=True):
    """Testa a política aprendida."""
    state = env.reset()
    state_idx = env.get_state_index(state)
    done = False
    total_reward = 0
    steps = 0
    
    if render:
        env.render(q_table=q_table, show_policy=True)
        time.sleep(1)
    
    while not done:
        # Selecionar a melhor ação da Q-table
        action = np.argmax(q_table[state_idx])
        
        # Executar ação
        next_state, reward, done = env.step(action)
        next_state_idx = env.get_state_index(next_state)
        
        if render:
            print(f"Estado: {state}, Ação: {env.action_names[action]}, Recompensa: {reward:.2f}")
            env.render(q_table=q_table, show_policy=True)
            time.sleep(0.5)
        
        # Atualizar estado
        state = next_state
        state_idx = next_state_idx
        
        total_reward += reward
        steps += 1
    
    print(f"\nTeste concluído:")
    print(f"Recompensa total: {total_reward:.2f}")
    print(f"Passos: {steps}")
    
    return total_reward, steps

def plot_results(rewards, steps):
    """Plota os resultados do treinamento."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Gráfico de recompensas
    ax1.plot(rewards, label='Recompensa por episódio')
    ax1.set_xlabel('Episódio')
    ax1.set_ylabel('Recompensa total')
    ax1.set_title('Recompensas durante o treinamento')
    ax1.legend()
    
    # Aplicar média móvel para suavizar a curva
    window_size = min(50, len(rewards))
    rewards_smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    ax1.plot(range(window_size-1, len(rewards)), rewards_smoothed, 'r-', label=f'Média móvel ({window_size} episódios)')
    ax1.legend()
    
    # Gráfico de passos
    ax2.plot(steps, label='Passos por episódio')
    ax2.set_xlabel('Episódio')
    ax2.set_ylabel('Número de passos')
    ax2.set_title('Passos necessários para atingir o objetivo')
    ax2.legend()
    
    # Aplicar média móvel para suavizar a curva
    steps_smoothed = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
    ax2.plot(range(window_size-1, len(steps)), steps_smoothed, 'r-', label=f'Média móvel ({window_size} episódios)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_q_values(env, q_table):
    """Visualiza os valores Q em uma grade."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Preparar os dados para o heatmap
    q_values = np.zeros((env.height, env.width))
    
    for i in range(env.height):
        for j in range(env.width):
            state_idx = env.get_state_index((i, j))
            if (i, j) == env.goal_pos:
                q_values[i, j] = 1  # Valor alto para o objetivo
            elif (i, j) in env.obstacles:
                q_values[i, j] = -1  # Valor baixo para obstáculos
            else:
                # Usar o valor Q máximo para esta célula
                q_values[i, j] = np.max(q_table[state_idx])
    
    # Definir cores personalizadas
    cmap = plt.cm.viridis
    
    # Criar o heatmap
    im = ax.imshow(q_values, cmap=cmap)
    
    # Configurar o colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Valor Q máximo')
    
    # Configurar os eixos
    ax.set_xticks(np.arange(env.width))
    ax.set_yticks(np.arange(env.height))
    ax.set_xticklabels(np.arange(env.width))
    ax.set_yticklabels(np.arange(env.height))
    
    # Adicionar valores e setas de política
    for i in range(env.height):
        for j in range(env.width):
            state_idx = env.get_state_index((i, j))
            if (i, j) not in env.obstacles:
                best_action = np.argmax(q_table[state_idx])
                arrow = ["↑", "→", "↓", "←"][best_action]
                ax.text(j, i, f"{arrow}\n{q_values[i, j]:.2f}", ha="center", va="center", color="w" if q_values[i, j] < 0.5 else "black")
    
    # Adicionar marcações especiais
    start_i, start_j = env.start_pos
    goal_i, goal_j = env.goal_pos
    
    # Marcar posição inicial e objetivo com círculos
    ax.add_patch(plt.Circle((start_j, start_i), 0.3, color='blue'))
    ax.add_patch(plt.Circle((goal_j, goal_i), 0.3, color='green'))
    
    # Titulo
    ax.set_title('Visualização dos Valores Q e Política')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Definir um ambiente Grid World com alguns obstáculos
    obstacles = [(1, 1), (1, 2), (2, 1), (3, 3)]
    env = GridWorld(height=5, width=5, start_pos=(0, 0), goal_pos=(4, 4), obstacles=obstacles)
    
    # Visualizar o ambiente inicial
    print("Ambiente inicial:")
    env.render()
    
    # Treinar o agente
    print("\nIniciando treinamento com Q-Learning...")
    q_table, rewards, steps = q_learning(env, num_episodes=500, alpha=0.1, gamma=0.99, 
                                         epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.995)
    
    # Visualizar resultados do treinamento
    plot_results(rewards, steps)
    
    # Visualizar a Q-table final
    print("\nVisualizando os valores Q aprendidos e a política resultante:")
    visualize_q_values(env, q_table)
    
    # Testar a política aprendida
    print("\nTestando a política aprendida:")
    test_policy(env, q_table) 