import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

class GridWorldVisualizer:
    def __init__(self, height=5, width=5, start_pos=(0, 0), goal_pos=(4, 4), obstacles=None):
        self.height = height
        self.width = width
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = [] if obstacles is None else obstacles
        
        # Ações possíveis: 0: cima, 1: direita, 2: baixo, 3: esquerda
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ["Cima", "Direita", "Baixo", "Esquerda"]
        self.action_arrows = ["↑", "→", "↓", "←"]
        
        # Inicializar Q-table com zeros
        self.num_states = height * width
        self.num_actions = len(self.actions)
        self.q_table = np.zeros((self.num_states, self.num_actions))
        
        # Parâmetros do Q-Learning
        self.alpha = 0.1  # Taxa de aprendizado
        self.gamma = 0.99  # Fator de desconto
        self.epsilon = 1.0  # Parâmetro de exploração
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        
        # Estado atual do agente
        self.agent_pos = start_pos
        
        # Histórico para animação
        self.history = []
        self.episode_count = 0
        
    def get_state_index(self, pos):
        return pos[0] * self.width + pos[1]
    
    def get_pos_from_index(self, index):
        return (index // self.width, index % self.width)
    
    def initialize_animation(self):
        self.fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(1, 2, width_ratios=[1, 1])
        
        # Grid World
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax1.set_xlim(0, self.width)
        self.ax1.set_ylim(0, self.height)
        self.ax1.set_xticks(np.arange(0, self.width + 1, 1))
        self.ax1.set_yticks(np.arange(0, self.height + 1, 1))
        self.ax1.grid(True)
        self.ax1.set_aspect('equal')
        
        # Inverter o eixo y para que (0,0) fique no canto superior esquerdo
        self.ax1.set_ylim(self.height, 0)
        
        # Desenhar o ambiente fixo (obstáculos, início, objetivo)
        self.draw_environment()
        
        # Q-Values
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax2.set_xlim(0, self.width)
        self.ax2.set_ylim(0, self.height)
        self.ax2.set_xticks(np.arange(0, self.width + 1, 1))
        self.ax2.set_yticks(np.arange(0, self.height + 1, 1))
        self.ax2.grid(True)
        self.ax2.set_aspect('equal')
        self.ax2.set_ylim(self.height, 0)
        
        # Título da animação
        self.ax1.set_title("Trajetória do Agente")
        self.ax2.set_title("Valores Q e Política")
        
        # Criar elementos que serão atualizados
        self.agent_circle = plt.Circle((self.start_pos[1] + 0.5, self.start_pos[0] + 0.5), 0.3, color='blue')
        self.ax1.add_patch(self.agent_circle)
        
        # Info text
        self.episode_text = self.ax1.text(0.05, -0.1, "", transform=self.ax1.transAxes)
        self.step_text = self.ax1.text(0.5, -0.1, "", transform=self.ax1.transAxes)
        self.reward_text = self.ax1.text(0.8, -0.1, "", transform=self.ax1.transAxes)
        
        # Desenhar q_values iniciais
        self.q_texts = {}
        self.policy_arrows = {}
        self.update_q_values()
        
        plt.tight_layout()
        
    def draw_environment(self):
        # Destacar a posição inicial
        start_j, start_i = self.start_pos[1], self.start_pos[0]
        start_rect = patches.Rectangle((start_j, start_i), 1, 1, linewidth=1, edgecolor='black', facecolor='yellow', alpha=0.3)
        self.ax1.add_patch(start_rect)
        self.ax1.text(start_j + 0.5, start_i + 0.5, "S", ha='center', va='center', fontweight='bold')
        
        # Destacar a posição do objetivo
        goal_j, goal_i = self.goal_pos[1], self.goal_pos[0]
        goal_rect = patches.Rectangle((goal_j, goal_i), 1, 1, linewidth=1, edgecolor='black', facecolor='green', alpha=0.3)
        self.ax1.add_patch(goal_rect)
        self.ax1.text(goal_j + 0.5, goal_i + 0.5, "G", ha='center', va='center', fontweight='bold')
        
        # Destacar obstáculos
        for obstacle in self.obstacles:
            obs_i, obs_j = obstacle
            obs_rect = patches.Rectangle((obs_j, obs_i), 1, 1, linewidth=1, edgecolor='black', facecolor='black')
            self.ax1.add_patch(obs_rect)
    
    def update_q_values(self):
        # Limpar textos anteriores
        for text in self.q_texts.values():
            text.remove()
        self.q_texts = {}
        
        for arrow in self.policy_arrows.values():
            arrow.remove()
        self.policy_arrows = {}
        
        # Desenhar o ambiente fixo no painel de valores Q
        self.ax2.clear()
        self.ax2.set_xlim(0, self.width)
        self.ax2.set_ylim(self.height, 0)
        self.ax2.set_xticks(np.arange(0, self.width + 1, 1))
        self.ax2.set_yticks(np.arange(0, self.height + 1, 1))
        self.ax2.grid(True)
        self.ax2.set_title("Valores Q e Política")
        
        # Desenhar obstáculos
        for obstacle in self.obstacles:
            obs_i, obs_j = obstacle
            obs_rect = patches.Rectangle((obs_j, obs_i), 1, 1, linewidth=1, edgecolor='black', facecolor='black')
            self.ax2.add_patch(obs_rect)
        
        # Destacar a posição inicial
        start_j, start_i = self.start_pos[1], self.start_pos[0]
        start_rect = patches.Rectangle((start_j, start_i), 1, 1, linewidth=1, edgecolor='black', facecolor='yellow', alpha=0.3)
        self.ax2.add_patch(start_rect)
        
        # Destacar a posição do objetivo
        goal_j, goal_i = self.goal_pos[1], self.goal_pos[0]
        goal_rect = patches.Rectangle((goal_j, goal_i), 1, 1, linewidth=1, edgecolor='black', facecolor='green', alpha=0.3)
        self.ax2.add_patch(goal_rect)
        
        # Adicionar valores Q e política
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) in self.obstacles:
                    continue
                    
                state_idx = self.get_state_index((i, j))
                best_action = np.argmax(self.q_table[state_idx])
                best_value = self.q_table[state_idx, best_action]
                
                # Mostrar o valor Q máximo
                if (i, j) == self.goal_pos:
                    text = self.ax2.text(j + 0.5, i + 0.5, "G", ha='center', va='center', fontweight='bold')
                else:
                    text = self.ax2.text(j + 0.5, i + 0.7, f"{best_value:.2f}", ha='center', va='center', fontsize=8)
                self.q_texts[(i, j)] = text
                
                # Mostrar a política (seta da melhor ação)
                if (i, j) != self.goal_pos:
                    arrow_text = self.action_arrows[best_action]
                    arrow = self.ax2.text(j + 0.5, i + 0.3, arrow_text, ha='center', va='center', fontsize=14, 
                                        color='red' if best_value > 0 else 'blue')
                    self.policy_arrows[(i, j)] = arrow
    
    def simulate_step(self, episode, step, state, action, reward, next_state, done):
        """Registra um passo da simulação para animação posterior"""
        self.history.append({
            'episode': episode,
            'step': step,
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'q_table': self.q_table.copy(),
            'done': done,
            'epsilon': self.epsilon
        })
    
    def update_animation(self, frame):
        if frame >= len(self.history):
            return
            
        data = self.history[frame]
        
        # Atualizar posição do agente
        i, j = data['state']
        self.agent_circle.center = (j + 0.5, i + 0.5)
        
        # Atualizar textos informativos
        self.episode_text.set_text(f"Episódio: {data['episode']+1}")
        self.step_text.set_text(f"Passo: {data['step']+1}")
        self.reward_text.set_text(f"Recompensa: {data['reward']:.2f}")
        
        # Atualizar Q-values
        self.q_table = data['q_table']
        self.update_q_values()
        
        # Destacar ação tomada
        if not data['done']:
            action = data['action']
            next_i, next_j = data['next_state']
            
            # Desenhar seta da ação tomada
            dx, dy = self.actions[action]
            # Use 0.4 como comprimento para evitar encher a célula
            arrow = patches.FancyArrow(j + 0.5, i + 0.5, 
                                     dx * 0.4, dy * 0.4, 
                                     width=0.05, head_width=0.2, head_length=0.2, 
                                     fc='blue', ec='blue')
            self.ax1.add_patch(arrow)
            
            # Remover a seta após 1 quadro
            if frame > 0:
                prev_data = self.history[frame-1]
                prev_i, prev_j = prev_data['state']
                for patch in self.ax1.patches:
                    if isinstance(patch, patches.FancyArrow):
                        patch.remove()
    
    def run_q_learning(self, num_episodes=10, max_steps=100, visualize_interval=1):
        """Executa o algoritmo Q-Learning e salva o histórico para animação"""
        self.history = []
        
        for episode in range(num_episodes):
            # Resetar ambiente
            self.agent_pos = self.start_pos
            state = self.agent_pos
            state_idx = self.get_state_index(state)
            total_reward = 0
            done = False
            
            # Registrar estado inicial
            if episode % visualize_interval == 0:
                self.simulate_step(episode, 0, state, -1, 0, state, False)
            
            for step in range(max_steps):
                # Política epsilon-greedy
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.num_actions)  # Exploração
                else:
                    action = np.argmax(self.q_table[state_idx])  # Exploração
                
                # Calcular próximo estado
                move = self.actions[action]
                next_state = (state[0] + move[0], state[1] + move[1])
                
                # Verificar se o movimento é válido
                if (0 <= next_state[0] < self.height and 
                    0 <= next_state[1] < self.width and 
                    next_state not in self.obstacles):
                    self.agent_pos = next_state
                else:
                    # Se movimento inválido, permanece no mesmo lugar
                    next_state = state
                
                # Calcular recompensa
                if self.agent_pos == self.goal_pos:
                    reward = 1.0
                    done = True
                else:
                    reward = -0.01
                
                next_state_idx = self.get_state_index(next_state)
                
                # Atualizar Q-value
                old_value = self.q_table[state_idx, action]
                next_max = np.max(self.q_table[next_state_idx])
                
                # Equação Q-Learning
                new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
                self.q_table[state_idx, action] = new_value
                
                # Registrar este passo para visualização
                if episode % visualize_interval == 0:
                    self.simulate_step(episode, step, state, action, reward, next_state, done)
                
                # Atualizar estado
                state = next_state
                state_idx = next_state_idx
                total_reward += reward
                
                if done:
                    break
            
            # Reduzir epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Registrar episódio
            if episode % 20 == 0:
                print(f"Episódio {episode+1}/{num_episodes}, Passos: {step+1}, Recompensa: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}")
            
            self.episode_count += 1
    
    def create_animation(self, filename=None, interval=200, fig_size=(16, 8)):
        """Cria uma animação a partir do histórico registrado"""
        self.initialize_animation()
        
        anim = animation.FuncAnimation(self.fig, self.update_animation, 
                                     frames=len(self.history), 
                                     interval=interval, 
                                     blit=False, repeat=False)
        
        if filename:
            anim.save(filename, writer='pillow', fps=5)
            print(f"Animação salva como {filename}")
        
        plt.tight_layout()
        plt.show()
        
        return anim

# Demonstração
if __name__ == "__main__":
    # Configurar o ambiente
    obstacles = [(1, 1), (1, 2), (2, 1), (3, 3)]
    vis = GridWorldVisualizer(height=5, width=5, start_pos=(0, 0), goal_pos=(4, 4), obstacles=obstacles)
    
    # Executar Q-Learning por alguns episódios, salvando estados para animação
    print("Executando Q-Learning...")
    vis.run_q_learning(num_episodes=50, max_steps=100, visualize_interval=10)
    
    # Criar animação
    print("Criando animação...")
    vis.create_animation(filename="q_learning_animation.gif", interval=500)
    
    # Mostrar política final
    print("\nPolítica final:")
    vis.initialize_animation()
    vis.update_q_values()
    plt.tight_layout()
    plt.show() 