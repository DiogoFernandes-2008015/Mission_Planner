import numpy as np
import matplotlib.pyplot as plt
import time

# =============================================================================
# FUNÇÕES AUXILIARES ADAPTADAS E VETORIZADAS
# =============================================================================

def gerpop(n, p, linf, lsup):
    """Inicialização aleatória uniforme genérica entre os limites para Rosenbrock."""
    # Gera uma matriz de formato (p, n) diretamente no intervalo [linf, lsup]
    return np.random.uniform(linf, lsup, size=(p, n))


def gervel(n, p, linf, lsup):
    """Inicialização aleatória de velocidades para o espaço de busca."""
    return np.random.uniform(linf, lsup, size=(p, n))


#def fitness(M, n, p):
#    """
#    Cálculo adaptado para a Função de Rosenbrock (Totalmente Vetorizado).
#    M tem dimensões (p, n), onde p é o nº de variáveis e n é o nº de partículas.
#    """
#    # xj representa os elementos de 1 até p-1 (índices 0 até p-2)
#    xj = M[:-1, :]
#    # xj1 representa os elementos de 2 até p (índices 1 até p-1)
#    xj1 = M[1:, :]
#    
#    # Aplica a fórmula matemática de Rosenbrock para todas as partículas de uma vez
#    termos = 100 * (xj**2 - xj1)**2 + (xj - 1)**2
#    
#    # Soma os resultados ao longo do eixo das variáveis (linhas), restando um vetor (n,)
#    aptd = np.sum(termos, axis=0)
#    return aptd

def fitness(M, n, p):
    """
    Cálculo adaptado para a Função de Dixon e Price (Totalmente Vetorizado).
    
    M : matriz populacional de dimensões (p, n)
    n : quantidade de partículas (colunas)
    p : quantidade de variáveis/dimensões (linhas)
    """
    # x(1) no MATLAB é o índice 0 no Python: (x_1 - 1)^2
    termo_inicial = (M[0, :] - 1) ** 2
    
    # xj representa os elementos de x(2) até x(p) -> índices 1 até p-1
    xj = M[1:, :]
    
    # xj_menos_1 representa os elementos de x(1) até x(p-1) -> índices 0 até p-2
    xj_menos_1 = M[:-1, :]
    
    # Criamos o vetor de multiplicadores j (de 2 até p)
    # Mudamos o formato para (p-1, 1) para aplicar broadcasting contra a matriz (p-1, n)
    j = np.arange(2, p + 1)[:, np.newaxis]
    
    # Calcula os termos do produtório/somatório de uma só vez
    termos_somatorio = j * (2 * (xj ** 2) - xj_menos_1) ** 2
    
    # Soma ao longo do eixo das variáveis (linhas), restando um vetor unidimensional (n,)
    s1 = np.sum(termos_somatorio, axis=0)
    
    # O fitness final é a soma do termo inicial com o resultado do somatório
    aptd = s1 + termo_inicial
    return aptd


def atv(V, M, P, Pg, alfa, beta, n, p):
    """Atualização da velocidade das partículas."""
    d1 = P - M
    d2 = Pg[:, np.newaxis] - M  # Expansão de dimensões para broadcasting automático
    
    r1 = np.random.uniform(0, 1)
    r2 = np.random.uniform(0, 1)
    
    return alfa * V + beta * (r1 * d1 + r2 * d2)


def reinipop(M, n, p):
    """Reinicialização aleatória de populações estagnadas."""
    return M * (1.0 - np.random.uniform(-1, 1, size=(p, n)))


# =============================================================================
# FUNÇÃO PRINCIPAL DO ALGORITMO PSO
# =============================================================================

def executar_pso(n, p, alfa, beta, nmi, tol, vmin, linf, lsup, k=1, count=0):
    """Executa o PSO focado em minimizar a função de Rosenbrock."""
    start_time = time.perf_counter()

    # Inicialização das matrizes de posição e velocidade
    M = gerpop(n, p, linf, lsup)
    V = np.zeros((p, n))

    # Cálculo do fitness inicial
    aptd = fitness(M, n, p)

    # Histórico para o gráfico dinâmico
    MAP = []
    k_inicial = k

    # Configuração da janela gráfica iterativa
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-') # Linha vermelha para destacar
    ax.grid(True)
    ax.set_title(f"Minimização da Função de Rosenbrock ({p} Variáveis)")
    ax.set_xlabel("Iterações")
    ax.set_ylabel("Melhor Fitness Global (Pgr)")

    # Loop do processo iterativo
    while k < nmi:
        
        # Atualização de P e Pr
        if k == k_inicial:
            P = M.copy()
            Pr = aptd.copy()
        else:
            melhorou = aptd < Pr
            if np.any(melhorou):
                P[:, melhorou] = M[:, melhorou]
                Pr[melhorou] = aptd[melhorou]
                
        # Atualização do melhor valor global
        posmin = np.argmin(Pr)
        Pgr = Pr[posmin]
        Pg = M[:, posmin]
        
        # Verificação da convergência (O mínimo global conhecido da Rosenbrock é 0.0)
        if abs(Pgr - vmin) < tol:
            break
            
        # Atualização populacional para a próxima iteração
        V = atv(V, M, P, Pg, alfa, beta, n, p)
        M = M + V
        
        # Verificação da estagnação do Método
        aptd = fitness(M, n, p)
        Pgr1 = np.min(aptd)
        print(f"Iteração {k} -> Melhor Atual (Pgr1) = {Pgr1:.6f}")
        
        if abs(Pgr - Pgr1) < tol:
            count += 1
        else:
            count = 0
            
        # Reinicialização se o algoritmo estagnar
        if count == 10:
            M = reinipop(M, n, p)
            count = 0
            
        MAP.append(Pgr)
        
        # Atualização rápida do gráfico dinâmico
        line.set_xdata(np.arange(len(MAP)))
        line.set_ydata(MAP)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)
        
        k += 1

    end_time = time.perf_counter()
    print(f"\nTempo de execução total: {end_time - start_time:.4f} segundos")

    plt.ioff()
    plt.show()
    
    return Pg, Pgr


# =============================================================================
# BLOCO DE CONFIGURAÇÃO DO USUÁRIO
# =============================================================================
if __name__ == "__main__":
    
    # Configuração padrão para os testes da Função de Rosenbrock
    configuracoes = {
        "n": 1000,          # População de partículas
        "p": 10,           # Número de variáveis (n do seu comentário original do Matlab)
        "alfa": 0.5,      # Fator de inércia
        "beta": 1.5,      # Fator populacional
        "nmi": 2000,       # Máximo de iterações
        "tol": 1e-7,      # Tolerância estrita para chegar perto de 0
        "vmin": 0.0,      # O mínimo global teórico de Rosenbrock é EXATAMENTE 0.0
        "linf": -10.0,     # Limite inferior padrão de testes da Rosenbrock
        "lsup": 10.0,      # Limite superior padrão de testes da Rosenbrock
    }
    
    print("Executando PSO Otimizado para a Função de Rosenbrock...")
    melhor_posicao, melhor_custo = executar_pso(**configuracoes)
    
    print("\n--- RESULTADOS FINAIS ---")
    print(f"Melhor Posição Encontrada (Pg): {melhor_posicao}") 
    print(f"Valor da Função no Ponto (Pgr): {melhor_custo:.10f} (Alvo ideal: 0.0)")