#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <array>

constexpr double PI = 3.14159265358979323846;
constexpr double g = 9.81;
constexpr double m = 10.0;
constexpr double S = PI * 0.105 * 0.105 / 4.0;
constexpr double cota = 0.0; // Cota de parada (solo)

struct State {
    double x, y, z;
    double vx, vy, vz;

    // Sobrecarga de operadores para facilitar o Runge-Kutta
    State operator+(const State& other) const {
        return {x + other.x, y + other.y, z + other.z, 
                vx + other.vx, vy + other.vy, vz + other.vz};
    }
    State operator*(double scalar) const {
        return {x * scalar, y * scalar, z * scalar, 
                vx * scalar, vy * scalar, vz * scalar};
    }
};

double get_atmosphere(double z) {
    constexpr double T0 = 288.15, P0 = 101325.0, L = 0.0065, R = 287.05;
    double z_clip = std::max(z, 0.0);
    double T = T0 - L * z_clip;
    double P = P0 * std::pow(1.0 - L * z_clip / T0, g / (R * L));
    return P / (R * T);
}

double get_cd(double mach) {
    const std::array<double, 7> mach_data = {0.0, 0.5, 0.8, 1.0, 1.2, 2.0, 3.0};
    const std::array<double, 7> cd_data   = {0.2, 0.2, 0.22, 0.4, 0.45, 0.35, 0.3};
    
    if (mach <= mach_data.front()) return cd_data.front();
    if (mach >= mach_data.back()) return cd_data.back();
    
    for (size_t i = 0; i < mach_data.size() - 1; ++i) {
        if (mach >= mach_data[i] && mach <= mach_data[i+1]) {
            double t = (mach - mach_data[i]) / (mach_data[i+1] - mach_data[i]);
            return cd_data[i] + t * (cd_data[i+1] - cd_data[i]);
        }
    }
    return 0.2;
}

std::array<double, 3> get_wind(double z) {
    constexpr double z_ref = 10.0;
    constexpr double alpha = 0.143;
    constexpr double wind_x_ref = 50.0;
    constexpr double wind_y_ref = 50.0;
    constexpr double wind_z_ref = 0.0;
    
    double z_clip = std::max(z, 0.001);
    double fator = std::pow(z_clip / z_ref, alpha);
    
    if (z <= 0.0) fator = 0.0;
    
    return {wind_x_ref * fator, wind_y_ref * fator, wind_z_ref};
}

State eqdif(const State& s) {
    std::array<double, 3> v_wind = get_wind(s.z);
    
    double v_rel_x = s.vx - v_wind[0];
    double v_rel_y = s.vy - v_wind[1];
    double v_rel_z = s.vz - v_wind[2];
    
    double v_rel_norm = std::sqrt(v_rel_x*v_rel_x + v_rel_y*v_rel_y + v_rel_z*v_rel_z);
    
    double mach = v_rel_norm / 340.0;
    double rho = get_atmosphere(s.z);
    double Cd = get_cd(mach);
    
    double drag_coeff = -0.5 * rho * S * Cd * v_rel_norm;
    
    double ax = (drag_coeff * v_rel_x) / m;
    double ay = (drag_coeff * v_rel_y) / m;
    double az = (-m * g + drag_coeff * v_rel_z) / m;
    
    return {s.vx, s.vy, s.vz, ax, ay, az};
}

std::pair<double, double> calcula_ponto_impacto(double x0, double y0, double z0, double v0, double theta_graus) {
    double theta_rad = theta_graus * PI / 180.0;
    State state = {x0, y0, z0, v0 * std::cos(theta_rad), v0 * std::sin(theta_rad), 0.0};
    
    double dt = 0.05; 
    
    // Loop de integração
    while (true) {
        State k1 = eqdif(state);
        State k2 = eqdif(state + k1 * (dt / 2.0));
        State k3 = eqdif(state + k2 * (dt / 2.0));
        State k4 = eqdif(state + k3 * dt);
        
        State next_state = state + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
        
        // Condição de parada: cruzou a cota Z=0 descendo
        if (next_state.z <= cota && next_state.vz < 0.0) {
            // Interpolação linear para encontrar exatamente o ponto de impacto z=0
            double t_interp = state.z / (state.z - next_state.z);
            double x_final = state.x + t_interp * (next_state.x - state.x);
            double y_final = state.y + t_interp * (next_state.y - state.y);
            return {x_final, y_final};
        }
        
        state = next_state;
        
        // Fallback de segurança caso a simulação exploda
        if (state.z < -100.0) break;
    }
    return {state.x, state.y};
}

struct Particle {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> pbest_pos;
    double pbest_fitness;
};

double fitness_balistica(const std::vector<double>& pos, double xd, double yd, double z0) {
    auto [x_final, y_final] = calcula_ponto_impacto(pos[0], pos[1], z0, pos[2], pos[3]);
    double dx = x_final - xd;
    double dy = y_final - yd;
    return dx*dx + dy*dy;
}

double get_rand(double min, double max, std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(gen);
}

void executar_pso_balistica(int n, int p, double alfa, double beta, int nmi, double tol, double vmin, 
                            const std::vector<double>& linf, const std::vector<double>& lsup, 
                            double xd, double yd, double z0) {
                            
    auto start_time = std::chrono::high_resolution_clock::now();
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<Particle> swarm(n);
    std::vector<double> gbest_pos(p);
    double gbest_fitness = std::numeric_limits<double>::max();

    // Inicialização da População
    for (int i = 0; i < n; ++i) {
        swarm[i].position.resize(p);
        swarm[i].velocity.resize(p, 0.0);
        swarm[i].pbest_pos.resize(p);
        
        for (int j = 0; j < p; ++j) {
            swarm[i].position[j] = get_rand(linf[j], lsup[j], gen);
        }
        
        swarm[i].pbest_fitness = fitness_balistica(swarm[i].position, xd, yd, z0);
        swarm[i].pbest_pos = swarm[i].position;
        
        if (swarm[i].pbest_fitness < gbest_fitness) {
            gbest_fitness = swarm[i].pbest_fitness;
            gbest_pos = swarm[i].position;
        }
    }

    int count = 0;
    double prev_gbest_fitness = gbest_fitness;

    // Loop de Otimização
    for (int k = 1; k <= nmi; ++k) {
        if (std::abs(gbest_fitness - vmin) < tol) break;

        for (int i = 0; i < n; ++i) {
            // Gera r1 e r2 uma vez por iteração/partícula (simulando a lógica original do Python)
            double r1 = get_rand(0.0, 1.0, gen);
            double r2 = get_rand(0.0, 1.0, gen);

            for (int j = 0; j < p; ++j) {
                // Atualiza Velocidade
                double d1 = swarm[i].pbest_pos[j] - swarm[i].position[j];
                double d2 = gbest_pos[j] - swarm[i].position[j];
                swarm[i].velocity[j] = alfa * swarm[i].velocity[j] + beta * (r1 * d1 + r2 * d2);
                
                // Atualiza Posição e Clip (limites)
                swarm[i].position[j] += swarm[i].velocity[j];
                swarm[i].position[j] = std::clamp(swarm[i].position[j], linf[j], lsup[j]);
            }

            // Avalia nova aptidão
            double current_fitness = fitness_balistica(swarm[i].position, xd, yd, z0);
            
            // Atualiza PBest
            if (current_fitness < swarm[i].pbest_fitness) {
                swarm[i].pbest_fitness = current_fitness;
                swarm[i].pbest_pos = swarm[i].position;
            }
            
            // Atualiza GBest
            if (current_fitness < gbest_fitness) {
                gbest_fitness = current_fitness;
                gbest_pos = swarm[i].position;
            }
        }

        //std::cout << "Iteração " << k << " -> Menor erro atual no enxame = " << std::fixed << std::setprecision(4) << gbest_fitness << " m^2\n";

        // Reinicialização do Enxame (Estagnação)
        if (std::abs(prev_gbest_fitness - gbest_fitness) < tol) {
            count++;
        } else {
            count = 0;
        }

        if (count == 10) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < p; ++j) {
                    double perturb = 1.0 - get_rand(-1.0, 1.0, gen);
                    swarm[i].position[j] = std::clamp(swarm[i].position[j] * perturb, linf[j], lsup[j]);
                }
            }
            count = 0;
        }
        prev_gbest_fitness = gbest_fitness;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    
    std::cout << "\nTempo de execução total: " << std::setprecision(4) << diff.count() << " segundos\n";
    std::cout << "\nOTIMIZAÇÃO CONCLUÍDA COM SUCESSO!\n";
    std::cout << "Menor Erro Quadrático Obtido: " << std::setprecision(6) << gbest_fitness << " m²\n";
    std::cout << "Distância linear até o alvo:  " << std::setprecision(3) << std::sqrt(gbest_fitness) << " metros\n";
    std::cout << "Parâmetros de Lançamento Ideais Encontrados:\n";
    std::cout << "  -> Coordenada X inicial (x0): " << std::setprecision(2) << gbest_pos[0] << " m\n";
    std::cout << "  -> Coordenada Y inicial (y0): " << std::setprecision(2) << gbest_pos[1] << " m\n";
    std::cout << "  -> Velocidade Inicial (v0):   " << std::setprecision(2) << gbest_pos[2] << " m/s\n";
    std::cout << "  -> Ângulo de Azimute (theta): " << std::setprecision(2) << gbest_pos[3] << "°\n";
}

// =============================================================================
// 6. EXECUÇÃO
// =============================================================================
int main() {
    double z0_const = 500.0;
    double xd_alvo = 500.0;
    double yd_alvo = 500.0;

    std::vector<double> lim_inf = {-10000.0, -10000.0, 10.0, 0.0};
    std::vector<double> lim_sup = { 10000.0,   1000.0, 10.0, 360.0};

    int n = 300;
    int p = 4;
    double alfa = 0.7;
    double beta = 1.4;
    int nmi = 1000;
    double tol = 1e-5;
    double vmin = 0.0;

    std::cout << "Iniciando Otimização de Lançamento (PSO + RK4 c/ Vento)...\n\n";

    executar_pso_balistica(n, p, alfa, beta, nmi, tol, vmin, lim_inf, lim_sup, xd_alvo, yd_alvo, z0_const);

    return 0;
}