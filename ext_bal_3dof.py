import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt

# ==========================================
# 1. Parâmetros e Constantes do Modelo
# ==========================================
g = 9.81
m = 10.0
S = jnp.pi * 0.105 * 0.105 / 4.0
cota = 0.0  # Cota de parada (solo)

# Dados de exemplo para o Coeficiente de Arrasto (Substitua pelos seus dados)
# Equivalente ao bloco "1-D T(u)" no Simulink
mach_data = jnp.array([0.0, 0.5, 0.8, 1.0, 1.2, 2.0, 3.0])
cd_data   = jnp.array([0.2, 0.2, 0.22, 0.4, 0.45, 0.35, 0.3])

# ==========================================
# 2. Funções Auxiliares (Atmosfera e Arrasto)
# ==========================================
def get_atmosphere(z):
    """
    Modelo de Atmosfera Padrão (ISA) simplificado para a troposfera.
    Substitui o bloco "ISA Atmosphere Model" do Simulink.
    """
    T0 = 288.15        # Temperatura no nível do mar [K]
    P0 = 101325.0      # Pressão no nível do mar [Pa]
    L = 0.0065         # Gradiente de temperatura [K/m]
    R = 287.05         # Constante dos gases [J/(kg.K)]
    
    z_clip = jnp.maximum(z, 0.0) # Evita valores negativos no subsolo
    T = T0 - L * z_clip
    P = P0 * (1.0 - L * z_clip / T0) ** (g / (R * L))
    rho = P / (R * T)
    
    return rho

def get_cd(mach):
    """
    Interpolação do coeficiente de arrasto baseado no Mach.
    """
    return jnp.interp(mach, mach_data, cd_data)

# ==========================================
# 3. Equação Diferencial (Vetor Field)
# ==========================================
def eqdif(t, state, args):
    """
    Equivalente à MATLAB Function eqdif e aos blocos de cálculo de Mach.
    Vetor de estado: [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state
    
    # Vetor velocidade e sua magnitude (norm(v))
    v = jnp.array([vx, vy, vz])
    v_norm = jnp.linalg.norm(v)
    
    # Cálculo do Número de Mach (Conforme constante 340 no Simulink)
    mach = v_norm / 340.0
    
    # Cálculo das propriedades atmosféricas e aerodinâmicas
    rho = get_atmosphere(z)
    Cd = get_cd(mach)
    
    # Forças (Equivalente às matrizes P e Drag no MATLAB)
    P_grav = jnp.array([0.0, 0.0, -m * g])
    Drag = -0.5 * rho * S * Cd * v_norm * v
    
    # Acelerações (App)
    acc = (1.0 / m) * (P_grav + Drag)
    
    # Retorna as derivadas: [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
    return jnp.array([vx, vy, vz, acc[0], acc[1], acc[2]])

# ==========================================
# 4. Condição de Parada (Nova API do Diffrax)
# ==========================================
def stop_condition(t, y, args, **kwargs):
    """
    Implementa a lógica: (z < cota) AND (vz < 0).
    A simulação para quando esta função retorna True.
    """
    z = y[2]
    vz = y[5]
    return jnp.logical_and(z < cota, vz < 0.0)

# ==========================================
# 5. Configuração e Execução do Solver Diffrax
# ==========================================
def simulate():
    # Condições iniciais: [x, y, z, vx, vy, vz]
    y0 = jnp.array([0.0, 0.0, 1000.0, 50.0, 0.0, 0.0]) 
    
    t0 = 0.0
    t1 = 100.0  # Tempo máximo de simulação
    dt0 = 0.01  # Passo inicial
    
    # Instanciando o problema ODE
    term = diffrax.ODETerm(eqdif)
    solver = diffrax.Dopri5() 
    
    # Controlador de passo adaptativo
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
    
    # Salvando a trajetória
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 1000))
    
    # Instanciando o evento de parada com a nova API
    event = diffrax.Event(stop_condition)
    
    # Resolução da EDO
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        event=event  # <-- Argumento atualizado aqui
    )
    
    return sol

# Executa a simulação
sol = simulate()
# ==========================================
# 6. Plotagem Rápida dos Resultados
# ==========================================
# Máscara para pegar apenas os dados até a parada (o Diffrax preenche o resto com o último valor)
valid_idx = sol.ts <= sol.ts[-1] 

t_vals = sol.ts[valid_idx]
x_vals = sol.ys[valid_idx, 0]
z_vals = sol.ys[valid_idx, 2]

plt.figure(figsize=(10, 5))
plt.plot(x_vals, z_vals, label="Trajetória (X vs Z)")
plt.axhline(y=cota, color='r', linestyle='--', label=f"Cota ({cota}m)")
plt.title("Trajetória Balística")
plt.xlabel("Distância X (m)")
plt.ylabel("Altitude Z (m)")
plt.legend()
plt.grid(True)
plt.show()
