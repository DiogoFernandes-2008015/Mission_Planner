import jax
import jax.numpy as jnp
import diffrax
import numpy as np
import matplotlib.pyplot as plt
import time

# =============================================================================
# 1. PARÂMETROS FÍSICOS E TRAJETÓRIA (JAX / DIFFRAX)
# =============================================================================
g = 9.81
m = 10.0
S = jnp.pi * 0.105 * 0.105 / 4.0
cota = 0.0  # Cota de parada (solo)

mach_data = jnp.array([0.0, 0.5, 0.8, 1.0, 1.2, 2.0, 3.0])
cd_data   = jnp.array([0.2, 0.2, 0.22, 0.4, 0.45, 0.35, 0.3])

def get_atmosphere(z):
    '''Modelo de Atmosfera Padrão (ISA) simplificado para a troposfera.
    '''
    T0, P0, L, R = 288.15, 101325.0, 0.0065, 287.05
    z_clip = jnp.maximum(z, 0.0)
    T = T0 - L * z_clip
    P = P0 * (1.0 - L * z_clip / T0) ** (g / (R * L))
    return P / (R * T)

def get_cd(mach):
    return jnp.interp(mach, mach_data, cd_data)

def eqdif(t, state, args):
    x, y, z, vx, vy, vz = state
    v = jnp.array([vx, vy, vz])
    v_norm = jnp.linalg.norm(v)
    mach = v_norm / 340.0
    rho = get_atmosphere(z)
    Cd = get_cd(mach)
    
    P_grav = jnp.array([0.0, 0.0, -m * g])
    Drag = -0.5 * rho * S * Cd * v_norm * v
    acc = (1.0 / m) * (P_grav + Drag)
    return jnp.array([vx, vy, vz, acc[0], acc[1], acc[2]])

def stop_condition(t, y, args, **kwargs):
    z, vz = y[2], y[5]
    return jnp.logical_and(z <= cota, vz < 0.0)

def calcula_ponto_impacto(x0, y0, z0, v0, theta_graus):
    """Calcula o ponto de impacto individual (z=0)"""
    theta_rad = theta_graus * jnp.pi / 180.0
    vx0 = v0 * jnp.cos(theta_rad)
    vy0 = v0 * jnp.sin(theta_rad)
    vz0 = 0.0
    
    y0_vec = jnp.array([x0, y0, z0, vx0, vy0, vz0]) 
    term = diffrax.ODETerm(eqdif)
    solver = diffrax.Dopri5() 
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
    
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=1000.0, dt0=0.01,
        y0=y0_vec, saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=stepsize_controller, event=diffrax.Event(stop_condition), max_steps=16384, throw=False
    )
    return sol.ys[0, 0], sol.ys[0, 1]

# Vetorização do JAX
calcula_ponto_impacto_vmap = jax.jit(jax.vmap(calcula_ponto_impacto, in_axes=(0, 0, None, 0, 0)))


# =============================================================================
# 2. FUNÇÃO DE APTIDÃO (FITNESS) ADAPTADA
# =============================================================================
def fitness_balistica(M, n, p, xd, yd, z0):
    """
    Calcula o erro de distância para todas as partículas simultaneamente.
    M possui dimensões (4, n):
      Linha 0: x0 | Linha 1: y0 | Linha 2: v0 | Linha 3: theta
    """
    x0_v = jnp.array(M[0, :])
    y0_v = jnp.array(M[1, :])
    v0_v = jnp.array(M[2, :])
    th_v = jnp.array(M[3, :])
    
    x_finais, y_finais = calcula_ponto_impacto_vmap(x0_v, y0_v, z0, v0_v, th_v)
    erros = (x_finais - xd)**2 + (y_finais - yd)**2
    
    return np.array(erros)


# =============================================================================
# 3. FUNÇÕES AUXILIARES DO PSO (ADAPTADAS PARA LIMITES VETORIAIS)
# =============================================================================
def gerpop(n, p, linf, lsup):
    return np.random.uniform(linf, lsup, size=(p, n))

def gervel(n, p, linf, lsup):
    return np.random.uniform(linf, lsup, size=(p, n))

def atv(V, M, P, Pg, alfa, beta, n, p):
    d1 = P - M
    d2 = Pg[:, np.newaxis] - M
    r1, r2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
    return alfa * V + beta * (r1 * d1 + r2 * d2)

def reinipop(M, n, p):
    return M * (1.0 - np.random.uniform(-1, 1, size=(p, n)))


# =============================================================================
# 4. ALGORITMO PSO PRINCIPAL
# =============================================================================
def executar_pso_balistica(n, p, alfa, beta, nmi, tol, vmin, linf, lsup, xd, yd, z0, k=1, count=0):
    start_time = time.perf_counter()

    M = gerpop(n, p, linf, lsup)
    V = np.zeros((p, n))

    aptd = fitness_balistica(M, n, p, xd, yd, z0)
    MAP = []
    k_inicial = k

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-')
    ax.grid(True)
    ax.set_title("Otimização de Parâmetros de Lançamento (PSO + Diffrax)")
    ax.set_xlabel("Iterações")
    ax.set_ylabel("Menor Erro Quadrático de Distância (m²)")

    while k < nmi:
        if k == k_inicial:
            P = M.copy()
            Pr = aptd.copy()
        else:
            melhorou = aptd < Pr
            if np.any(melhorou):
                P[:, melhorou] = M[:, melhorou]
                Pr[melhorou] = aptd[melhorou]
                
        posmin = np.argmin(Pr)
        Pgr = Pr[posmin]
        Pg = M[:, posmin]
        
        if abs(Pgr - vmin) < tol:
            break
            
        V = atv(V, M, P, Pg, alfa, beta, n, p)
        M = np.clip(M + V, linf, lsup)
        
        aptd = fitness_balistica(M, n, p, xd, yd, z0)
        Pgr1 = np.min(aptd)
        print(f"Iteração {k} -> Menor erro atual no enxame = {Pgr1:.4f} m²")
        
        if abs(Pgr - Pgr1) < tol:
            count += 1
        else:
            count = 0
            
        if count == 10:
            M = np.clip(reinipop(M, n, p), linf, lsup)
            count = 0
            
        MAP.append(Pgr)
        
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
# 5. CONFIGURAÇÃO DO PROBLEMA E EXECUÇÃO
# =============================================================================
if __name__ == "__main__":
    
    Z0_CONSTANTE = 500.0   
    XD_ALVO = 500.0  
    YD_ALVO = 500.0     
    
    LIMITES_INFERIORES = np.array([[-10000.0],  
                                   [-10000.0],   
                                   [10.0],    
                                   [0.0]])    

    LIMITES_SUPERIORES = np.array([[10000.0],   
                                   [1000.0],    
                                   [10.0],   
                                   [360.0]])   

    configuracoes = {
        "n": 300,              
        "p": 4,
        "alfa": 0.7,
        "beta": 1.4,
        "nmi": 100,
        "tol": 1e-1,
        "vmin": 0.0,
        "linf": LIMITES_INFERIORES,
        "lsup": LIMITES_SUPERIORES,
        "xd": XD_ALVO,
        "yd": YD_ALVO,
        "z0": Z0_CONSTANTE
    }
    
    melhores_parametros, menor_erro = executar_pso_balistica(**configuracoes)

    print(f"\n")
    print(f"OTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
    
    print(f"Menor Erro Quadrático Obtido: {menor_erro:.6f} m²")
    print(f"Distância linear até o alvo:  {np.sqrt(menor_erro):.3f} metros")
    
    print(f"Parâmetros de Lançamento Ideais Encontrados:")
    print(f"  -> Coordenada X inicial (x0): {melhores_parametros[0]:.2f} m")
    print(f"  -> Coordenada Y inicial (y0): {melhores_parametros[1]:.2f} m")
    print(f"  -> Velocidade Inicial (v0):   {melhores_parametros[2]:.2f} m/s")
    print(f"  -> Ângulo de Azimute (theta):  {melhores_parametros[3]:.2f}°")
    