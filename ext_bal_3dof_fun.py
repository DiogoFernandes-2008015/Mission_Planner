import jax
import jax.numpy as jnp
import diffrax

# ==========================================
# 1. Parâmetros Globais e Dados
# ==========================================
g = 9.81
m = 10.0
S = jnp.pi * 0.105 * 0.105 / 4.0
cota = 0.0  # Cota de parada (solo)

mach_data = jnp.array([0.0, 0.5, 0.8, 1.0, 1.2, 2.0, 3.0])
cd_data   = jnp.array([0.2, 0.2, 0.22, 0.4, 0.45, 0.35, 0.3])

# ==========================================
# 2. Funções Físicas
# ==========================================
def get_atmosphere(z):
    T0 = 288.15
    P0 = 101325.0
    L = 0.0065
    R = 287.05
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
    z = y[2]
    vz = y[5]
    return jnp.logical_and(z <= cota, vz < 0.0)

# ==========================================
# 3. Função Principal de Simulação
# ==========================================
@jax.jit
def calcula_ponto_impacto(x0, y0, z0, v0, theta_graus):
    """
    Calcula as coordenadas X e Y do ponto de impacto (z=0).
    
    Parâmetros:
    - x0, y0: Coordenadas horizontais iniciais (m)
    - z0: Altitude inicial (m)
    - v0: Módulo da velocidade inicial (m/s)
    - theta_graus: Ângulo de elevação com o eixo X (graus)
    
    Retorna:
    - x_final, y_final: Coordenadas no momento do impacto.
    """
    # Converte ângulo para radianos
    theta_rad = theta_graus * jnp.pi / 180.0
    
    # Decompõe a velocidade inicial (Assumindo movimento no plano X-Z)
    vx0 = v0 * jnp.cos(theta_rad)
    vy0 = v0 * jnp.sin(theta_rad)
    vz0 = 0
    
    # Vetor de estado inicial: [x, y, z, vx, vy, vz]
    y0_vec = jnp.array([x0, y0, z0, vx0, vy0, vz0]) 
    
    # Configurações de tempo
    t0 = 0.0
    t1 = 1000.0  # Tempo máximo folgado para garantir que atinja o solo
    dt0 = 0.01
    
    term = diffrax.ODETerm(eqdif)
    solver = diffrax.Dopri5() 
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
    
    # IMPORTANTE: Em vez de salvar toda a trajetória, salva apenas o momento final (t1 ou evento)
    saveat = diffrax.SaveAt(t1=True)
    event = diffrax.Event(stop_condition)
    
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0_vec,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        event=event
    )
    
    # Como salvamos apenas em t1, o array ys tem formato (1, 6). 
    # Extraímos x e y do único ponto salvo.
    x_final = sol.ys[0, 0]
    y_final = sol.ys[0, 1]
    
    return x_final, y_final

# ==========================================
# 4. Exemplo de Uso
# ==========================================
if __name__ == "__main__":
    # Entradas: Inicia em x=0, y=0, z=1000m, v=50m/s, ângulo=15 graus
    x_imp, y_imp = calcula_ponto_impacto(x0=0.0, y0=0.0, z0=1000.0, v0=50.0, theta_graus=60.0)
    
    print(f"Coordenada X do impacto: {x_imp:.2f} m")
    print(f"Coordenada Y do impacto: {y_imp:.2f} m")
