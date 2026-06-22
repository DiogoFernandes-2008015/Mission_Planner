#ifndef UNICODE
#define UNICODE
#endif

#include <windows.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <array>
#include <sstream>
#include <string>

// =============================================================================
// 1. PARÂMETROS FÍSICOS, ESTRUTURAS E LÓGICA DO MODELO
// =============================================================================
constexpr double PI = 3.14159265358979323846;
constexpr double g = 9.81;
constexpr double m = 10.0;
constexpr double S = PI * 0.105 * 0.105 / 4.0;

struct ModelParams {
    double cota;
    double z0_inicial; // Nova variável de entrada inserida via GUI
    double x_alvo;
    double y_alvo;
    double wind_x_ref;
    double wind_y_ref;
    double z_ref;
} params;

struct State {
    double x, y, z;
    double vx, vy, vz;
    State operator+(const State& other) const {
        return { x + other.x, y + other.y, z + other.z, vx + other.vx, vy + other.vy, vz + other.vz };
    }
    State operator*(double scalar) const {
        return { x * scalar, y * scalar, z * scalar, vx * scalar, vy * scalar, vz * scalar };
    }
};

double get_atmosphere(double z) {
    constexpr double T0 = 288.15, P0 = 101325.0, L = 0.0065, R = 287.05;
    double z_clip = std::max(z, 0.0);
    return (P0 * std::pow(1.0 - L * z_clip / T0, g / (R * L))) / (R * (T0 - L * z_clip));
}

double get_cd(double mach) {
    const std::array<double, 7> mach_data = { 0.0, 0.5, 0.8, 1.0, 1.2, 2.0, 3.0 };
    const std::array<double, 7> cd_data = { 0.2, 0.2, 0.22, 0.4, 0.45, 0.35, 0.3 };
    if (mach <= mach_data.front()) return cd_data.front();
    if (mach >= mach_data.back()) return cd_data.back();
    for (size_t i = 0; i < mach_data.size() - 1; ++i) {
        if (mach >= mach_data[i] && mach <= mach_data[i + 1]) {
            double t = (mach - mach_data[i]) / (mach_data[i + 1] - mach_data[i]);
            return cd_data[i] + t * (cd_data[i + 1] - cd_data[i]);
        }
    }
    return 0.2;
}

std::array<double, 3> get_wind(double z) {
    constexpr double alpha = 0.143;
    double z_clip = std::max(z, 0.001);
    double fator = std::pow(z_clip / (params.z_ref > 0 ? params.z_ref : 10.0), alpha);
    if (z <= 0.0) fator = 0.0;
    return { params.wind_x_ref * fator, params.wind_y_ref * fator, 0.0 };
}

State eqdif(const State& s) {
    std::array<double, 3> v_wind = get_wind(s.z);
    double v_rel_x = s.vx - v_wind[0], v_rel_y = s.vy - v_wind[1], v_rel_z = s.vz - v_wind[2];
    double v_rel_norm = std::sqrt(v_rel_x * v_rel_x + v_rel_y * v_rel_y + v_rel_z * v_rel_z);
    double drag_coeff = -0.5 * get_atmosphere(s.z) * S * get_cd(v_rel_norm / 340.0) * v_rel_norm;
    return { s.vx, s.vy, s.vz, (drag_coeff * v_rel_x) / m, (drag_coeff * v_rel_y) / m, (-m * g + drag_coeff * v_rel_z) / m };
}

std::pair<double, double> calcula_ponto_impacto(double x0, double y0, double z0, double v0, double theta_graus) {
    double theta_rad = theta_graus * PI / 180.0;
    State state = { x0, y0, z0, v0 * std::cos(theta_rad), v0 * std::sin(theta_rad), 0.0 };
    double dt = 0.05;
    while (true) {
        State k1 = eqdif(state), k2 = eqdif(state + k1 * (dt / 2.0)), k3 = eqdif(state + k2 * (dt / 2.0)), k4 = eqdif(state + k3 * dt);
        State next_state = state + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
        if (next_state.z <= params.cota && next_state.vz < 0.0) {
            double t_interp = (state.z - params.cota) / (state.z - next_state.z);
            return { state.x + t_interp * (next_state.x - state.x), state.y + t_interp * (next_state.y - state.y) };
        }
        state = next_state;
        if (state.z < (params.cota - 100.0)) break;
    }
    return { state.x, state.y };
}

struct Particle { std::vector<double> position, velocity, pbest_pos; double pbest_fitness; };

double fitness_balistica(const std::vector<double>& pos, double xd, double yd, double z0) {
    auto [x_final, y_final] = calcula_ponto_impacto(pos[0], pos[1], z0, pos[2], pos[3]);
    double dx = x_final - xd, dy = y_final - yd;
    return dx * dx + dy * dy;
}

std::wstring executar_pso_balistica_gui() {
    std::random_device rd; std::mt19937 gen(rd());
    int n = 200, p = 4, nmi = 400;
    double alfa = 0.7, beta = 1.4, tol = 1e-5, vmin = 0.0;
    std::vector<double> linf = { -10000.0, -10000.0, 10.0, 0.0 }, lsup = { 10000.0, 10000.0, 10.0, 360.0 };

    std::vector<Particle> swarm(n);
    std::vector<double> gbest_pos(p);
    double gbest_fitness = std::numeric_limits<double>::max();

    // Utiliza a cota de lançamento (Z0) inserida dinamicamente pelo usuário
    double z0_inicial = params.z0_inicial;

    for (int i = 0; i < n; ++i) {
        swarm[i].position = { std::uniform_real_distribution<double>(linf[0], lsup[0])(gen), std::uniform_real_distribution<double>(linf[1], lsup[1])(gen), std::uniform_real_distribution<double>(linf[2], lsup[2])(gen), std::uniform_real_distribution<double>(linf[3], lsup[3])(gen) };
        swarm[i].velocity.resize(p, 0.0);
        swarm[i].pbest_fitness = fitness_balistica(swarm[i].position, params.x_alvo, params.y_alvo, z0_inicial);
        swarm[i].pbest_pos = swarm[i].position;
        if (swarm[i].pbest_fitness < gbest_fitness) { gbest_fitness = swarm[i].pbest_fitness; gbest_pos = swarm[i].position; }
    }

    int count = 0; double prev_gbest_fitness = gbest_fitness;
    for (int k = 1; k <= nmi; ++k) {
        if (std::abs(gbest_fitness - vmin) < tol) break;
        for (int i = 0; i < n; ++i) {
            double r1 = std::uniform_real_distribution<double>(0.0, 1.0)(gen), r2 = std::uniform_real_distribution<double>(0.0, 1.0)(gen);
            for (int j = 0; j < p; ++j) {
                swarm[i].velocity[j] = alfa * swarm[i].velocity[j] + beta * (r1 * (swarm[i].pbest_pos[j] - swarm[i].position[j]) + r2 * (gbest_pos[j] - swarm[i].position[j]));
                swarm[i].position[j] = std::clamp(swarm[i].position[j] + swarm[i].velocity[j], linf[j], lsup[j]);
            }
            double current_fitness = fitness_balistica(swarm[i].position, params.x_alvo, params.y_alvo, z0_inicial);
            if (current_fitness < swarm[i].pbest_fitness) { swarm[i].pbest_fitness = current_fitness; swarm[i].pbest_pos = swarm[i].position; }
            if (current_fitness < gbest_fitness) { gbest_fitness = current_fitness; gbest_pos = swarm[i].position; }
        }
        if (std::abs(prev_gbest_fitness - gbest_fitness) < tol) count++; else count = 0;
        if (count == 10) {
            for (int i = 0; i < n; ++i) for (int j = 0; j < p; ++j) swarm[i].position[j] = std::clamp(swarm[i].position[j] * (1.0 - std::uniform_real_distribution<double>(-1.0, 1.0)(gen)), linf[j], lsup[j]);
            count = 0;
        }
        prev_gbest_fitness = gbest_fitness;
    }

    std::wstringstream ss;
    ss << std::fixed;
    ss << L"         RESULTADOS ENCONTRADOS\r\n";
    ss << std::setprecision(6);
    ss << L"Erro Quadrático : " << gbest_fitness << L" m²\r\n";
    ss << L"Precisão do Alvo: " << std::sqrt(gbest_fitness) << L" m de erro\r\n";

    ss << std::setprecision(2);
    ss << L"  -> X do Lançamento: " << gbest_pos[0] << L" m\r\n";
    ss << L"  -> Y do Lançamento: " << gbest_pos[1] << L" m\r\n";
    ss << L"  -> V do Lançamento: " << gbest_pos[2] << L" m/s\r\n";
    ss << L"  -> Azimute de Lançamento: " << gbest_pos[3] << L"°\r\n";
    return ss.str();
}

// =============================================================================
// 2. CONSTRUÇÃO DA INTERFACE GRÁFICA (WIN32 GUI)
// =============================================================================
HWND hCota, hZ0, hXAlvo, hYAlvo, hVx, hVy, hZref, hBtnCalc, hResult;

float ObterValorJanela(HWND hWnd) {
    wchar_t buffer[128];
    GetWindowTextW(hWnd, buffer, 128);
    std::wstring ws(buffer);
    std::replace(ws.begin(), ws.end(), L',', L'.'); // Trata vírgula como ponto
    try { return std::stod(ws); }
    catch (...) { return 0.0f; }
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CREATE: {
        int lblX = 20, inputX = 180, width = 100, height = 23, spacing = 28, currentY = 15;

        auto AdicionarComponente = [&](const wchar_t* texto, HWND& hInput) {
            CreateWindowW(L"STATIC", texto, WS_VISIBLE | WS_CHILD, lblX, currentY, inputX - 30, height, hwnd, NULL, NULL, NULL);
            hInput = CreateWindowW(L"EDIT", L"0.0", WS_VISIBLE | WS_CHILD | WS_BORDER | ES_AUTOHSCROLL, inputX, currentY, width, height, hwnd, NULL, NULL, NULL);
            currentY += spacing;
            };

        // Inputs na Interface
        AdicionarComponente(L"Cota do Alvo (m):", hCota);
        SetWindowTextW(hCota, L"0.0");

        AdicionarComponente(L"Altura de voo (m):", hZ0); 
        SetWindowTextW(hZ0, L"500.0");

        AdicionarComponente(L"X do Alvo (m):", hXAlvo);
        SetWindowTextW(hXAlvo, L"500.0");

        AdicionarComponente(L"Y do Alvo (m):", hYAlvo);
        SetWindowTextW(hYAlvo, L"500.0");

        AdicionarComponente(L"Vx do vento (m/s):", hVx);
        SetWindowTextW(hVx, L"50.0");

        AdicionarComponente(L"Vy do vento (m/s):", hVy);
        SetWindowTextW(hVy, L"50.0");

        AdicionarComponente(L"Zref do vento (m):", hZref);
        SetWindowTextW(hZref, L"10.0");

        // Botão Calcular
        hBtnCalc = CreateWindowW(L"BUTTON", L"Calcular Parâmetros", WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON, lblX, currentY + 5, 260, 35, hwnd, (HMENU)1, NULL, NULL);

        // Caixa Lateral de Resultados (Aumentada um pouco para comportar o novo leiaute)
        hResult = CreateWindowW(L"EDIT", L"Insira os dados e clique em Calcular...", WS_VISIBLE | WS_CHILD | WS_BORDER | ES_MULTILINE | ES_READONLY | WS_VSCROLL, 300, 15, 320, 245, hwnd, NULL, NULL, NULL);

        // Ajustar Fonte para uma mais moderna (Segoe UI)
        HFONT hFont = CreateFontW(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
        SendMessage(hwnd, WM_SETFONT, (WPARAM)hFont, TRUE);
        EnumChildWindows(hwnd, [](HWND child, LPARAM font) -> BOOL { SendMessage(child, WM_SETFONT, font, TRUE); return TRUE; }, (LPARAM)hFont);
        return 0;
    }
    case WM_COMMAND: {
        if (LOWORD(wParam) == 1) { // Clique no Botão Calcular
            SetWindowTextW(hResult, L"Calculando via PSO... Aguarde alguns instantes.");
            UpdateWindow(hResult);

            // Captura todos os dados digitados na tela, incluindo o novo Z0
            params.cota = ObterValorJanela(hCota);
            params.z0_inicial = ObterValorJanela(hZ0);
            params.x_alvo = ObterValorJanela(hXAlvo);
            params.y_alvo = ObterValorJanela(hYAlvo);
            params.wind_x_ref = ObterValorJanela(hVx);
            params.wind_y_ref = ObterValorJanela(hVy);
            params.z_ref = ObterValorJanela(hZref);

            // Executa a otimização balística e plota na tela
            std::wstring resultado = executar_pso_balistica_gui();
            SetWindowTextW(hResult, resultado.c_str());
        }
        return 0;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    const wchar_t CLASS_NAME[] = L"SimuladorBalisticoClass";
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);

    RegisterClassW(&wc);

    // Ajustado a altura da janela de 300 para 320 para acomodar confortavelmente o novo campo
    HWND hwnd = CreateWindowExW(0, CLASS_NAME, L"Simulador Balístico (PSO + GUI)", WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX, CW_USEDEFAULT, CW_USEDEFAULT, 660, 320, NULL, NULL, hInstance, NULL);
    if (hwnd == NULL) return 0;

    ShowWindow(hwnd, nCmdShow);

    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return 0;
}