# Q-Learning em Grid World

Este projeto implementa o algoritmo de aprendizado por reforço Q-Learning para resolver o problema clássico do Grid World.

## Autores

- Arthur Henrique Tscha Vieira
- Rafael Rodrigues Ferreira de Andrade

## Descrição

O Grid World é um ambiente em forma de grade onde um agente precisa navegar do ponto inicial até um objetivo, evitando obstáculos. Este projeto demonstra como o algoritmo Q-Learning pode ser usado para que o agente aprenda a encontrar o caminho ótimo através de tentativa e erro.

## Arquivos do Projeto

- `grid_world.py`: Implementação completa do ambiente Grid World e do algoritmo Q-Learning
- `visualizacao_q_learning.py`: Visualizador que cria animações para ilustrar o processo de aprendizagem
- `q_learning_teoria.md`: Explicação detalhada do modelo matemático do Q-Learning

## Conceito do Q-Learning

O Q-Learning é um algoritmo de aprendizado por reforço que permite a um agente aprender a tomar decisões ótimas através de interações com um ambiente. Ele aprende uma função de valor Q(s,a) que representa a utilidade esperada de tomar uma ação 'a' em um estado 's'.

A equação central do Q-Learning é:

Q(s,a) ← Q(s,a) + α _ [R + γ _ max Q(s',a') - Q(s,a)]

Onde:

- α (alpha): taxa de aprendizado
- γ (gamma): fator de desconto
- R: recompensa imediata
- max Q(s',a'): estimativa da recompensa futura máxima

## Como Executar

Para executar o algoritmo:

```bash
python grid_world.py
```

Para criar e visualizar animações do processo de aprendizagem:

```bash
python visualizacao_q_learning.py
```

## Requisitos

- Python 3.6+
- NumPy
- Matplotlib

## Resultados

Após o treinamento, o agente aprende a política ótima, que é o caminho mais curto do ponto inicial até o objetivo, evitando obstáculos. As visualizações mostram como os valores Q evoluem durante o treinamento e como a política melhora com o tempo.
