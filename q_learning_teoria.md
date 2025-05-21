# Aprendizado por Reforço: Q-Learning

## Conceito

O Q-Learning é um algoritmo de aprendizado por reforço que permite a um agente aprender a tomar decisões ótimas através de interações com um ambiente. É um método de **diferença temporal (TD)** que aprende o valor de uma ação em um determinado estado, sem necessidade de um modelo do ambiente.

As principais características do Q-Learning são:

- **Livre de modelo** (model-free): não requer conhecimento prévio da dinâmica do ambiente
- **Fora de política** (off-policy): pode aprender a política ótima independentemente da política que está sendo seguida
- **Baseado em valor** (value-based): aprende uma função de valor que estima a utilidade de tomar uma ação em um estado

## Modelo Matemático

### Função Q

O Q-Learning mantém uma tabela de valores Q, que representa a utilidade esperada de tomar uma ação específica em um estado específico. Essa tabela é chamada **Q-table**:

Q(s, a) = valor esperado ao tomar a ação 'a' no estado 's'

Onde:

- s é um estado
- a é uma ação possível nesse estado

### Equação de Bellman

A função Q ótima obedece à **Equação de Bellman**:

Q*(s, a) = R(s, a) + γ * max<sub>a'</sub> Q\*(s', a')

Onde:

- Q\*(s, a) é o valor Q ótimo para o par estado-ação (s, a)
- R(s, a) é a recompensa imediata por tomar a ação 'a' no estado 's'
- γ (gamma) é o fator de desconto (0 ≤ γ ≤ 1) que determina a importância das recompensas futuras
- s' é o próximo estado resultante de tomar a ação 'a' no estado 's'
- max<sub>a'</sub> Q\*(s', a') é o valor máximo possível no próximo estado 's'

### Algoritmo de Diferença Temporal (TD)

O Q-Learning usa uma abordagem de diferença temporal para atualizar iterativamente a função Q. A fórmula de atualização é:

Q(s, a) ← Q(s, a) + α _ [R + γ _ max<sub>a'</sub> Q(s', a') - Q(s, a)]

Onde:

- α (alpha) é a taxa de aprendizado (0 < α ≤ 1) que controla o quanto cada nova experiência sobrescreve o conhecimento anterior
- [R + γ * max<sub>a'</sub> Q(s', a') - Q(s, a)] é o **erro de diferença temporal**

Esta fórmula pode ser decomposta em:

1. **Recompensa imediata**: R
2. **Estimativa da recompensa futura**: γ \* max<sub>a'</sub> Q(s', a')
3. **Valor atual estimado**: Q(s, a)
4. **Erro TD**: [R + γ * max<sub>a'</sub> Q(s', a') - Q(s, a)]

## Passos do Algoritmo Q-Learning

1. **Inicialização**:

   - Inicializar a Q-table com zeros ou valores aleatórios
   - Definir os hiperparâmetros α (taxa de aprendizado), γ (fator de desconto) e ε (parâmetro de exploração)

2. **Para cada episódio**:

   - Inicializar o estado s

   - **Para cada passo do episódio**:

     - Escolher uma ação 'a' usando a política ε-greedy:

       - Com probabilidade ε: escolher uma ação aleatória (exploração)
       - Com probabilidade (1-ε): escolher a ação com maior valor Q (exploração)

     - Executar a ação 'a', observar a recompensa 'R' e o próximo estado 's''

     - Atualizar o valor Q:
       Q(s, a) ← Q(s, a) + α _ [R + γ _ max<sub>a'</sub> Q(s', a') - Q(s, a)]

     - Atualizar o estado: s ← s'

     - Se s é um estado terminal, encerrar o episódio

   - Reduzir gradualmente ε (para diminuir a exploração com o tempo)

3. **Final**:
   - Retornar a Q-table

## Ilustração: Resolução do Grid World

O Grid World é um ambiente clássico em aprendizado por reforço, onde um agente navega por uma grade para alcançar um objetivo.

### Exemplo de Grid World 5x5:

```
┌───┬───┬───┬───┬───┐
│ A │   │   │   │   │
├───┼───┼───┼───┼───┤
│   │ X │ X │   │   │
├───┼───┼───┼───┼───┤
│   │ X │   │   │   │
├───┼───┼───┼───┼───┤
│   │   │   │ X │   │
├───┼───┼───┼───┼───┤
│   │   │   │   │ G │
└───┴───┴───┴───┴───┘
```

Onde:

- A: Agente (posição inicial)
- G: Objetivo (Goal)
- X: Obstáculos

### Passos para solução com Q-Learning:

1. **Representação do ambiente**:

   - Estados: cada célula da grade (25 estados)
   - Ações: movimento em 4 direções (cima, direita, baixo, esquerda)
   - Recompensas: -0.01 por passo, +1.0 ao atingir o objetivo

2. **Inicialização da Q-table**:

   - Criar uma tabela 25×4 (25 estados × 4 ações) com valores zero

3. **Processo de aprendizagem**:

   - O agente explora o ambiente usando a política ε-greedy
   - Para cada movimento, a Q-table é atualizada usando a equação TD
   - Ao longo dos episódios, a política se torna cada vez mais ótima

4. **Resultado final**:
   - Após o treinamento, o agente descobre o caminho mais curto do ponto inicial até o objetivo, evitando obstáculos
   - A política ótima pode ser extraída da Q-table tomando a ação com o maior valor Q em cada estado

## Exemplo de Sequência de Aprendizagem (Simplificada)

**Episódio 1**: O agente se move aleatoriamente pela grade, eventualmente encontrando o objetivo por acaso. Os valores Q começam a ser atualizados.

**Episódio 10**: O agente começa a mostrar tendências para movimentos mais eficientes, mas ainda explora bastante.

**Episódio 100**: O agente já identificou os caminhos promissores e raramente toma ações subótimas.

**Episódio 500**: A política está quase convergida para a solução ótima, e o agente consistentemente escolhe o caminho mais curto para o objetivo.

## Convergência

A teoria do Q-Learning garante que, sob certas condições (visitação suficiente de todos os pares estado-ação e um cronograma apropriado de taxa de aprendizado), o algoritmo converge para a política ótima.

## Vantagens e Aplicações

- **Simples e eficaz**: Fácil de implementar e entender
- **Versatilidade**: Aplicável a uma ampla gama de problemas
- **Garantia teórica**: Converge para a política ótima sob condições apropriadas

O Q-Learning é aplicado em diversos campos, incluindo robótica, jogos, sistemas de recomendação, controle de tráfego, e muito mais.
