# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:47:13 2026

@author: HOME
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parámetros del espectro discreto
# ============================================================
N = 21
omega = np.linspace(20, 50, N)   # 21 frecuencias entre 20 y 50
omega0 = 35.0                    # frecuencia central
sigma = 6.3                      # anchura de la gaussiana espectral

# Amplitudes espectrales gaussianas
A = np.exp(-((omega - omega0)**2) / (2 * sigma**2))
A = A / A.max()

# Fases espectrales
phi_const = np.zeros_like(omega)

# IGUAL que en la figura anterior
beta2 = 0.08
phi_quad = beta2 * (omega - omega0)**2

# ============================================================
# Funciones
# ============================================================
def campo_complejo(t, omega, A, phi):
    return np.sum(
        A[:, None] * np.exp(1j * (omega[:, None] * t[None, :] + phi[:, None])),
        axis=0
    )

def campo_real(t, omega, A, phi):
    return np.real(campo_complejo(t, omega, A, phi))

def envolvente(t, omega, A, phi):
    return np.abs(campo_complejo(t, omega, A, phi))

def intensidad(t, omega, A, phi):
    E = campo_complejo(t, omega, A, phi)
    return np.abs(E)**2

def encontrar_maximos_locales(y):
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1

def encontrar_minimos_locales(y):
    return np.where((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]))[0] + 1

def seleccionar_par_de_minimos(t, y, t_ref, ventana):
    """
    Selecciona dos mínimos consecutivos dentro de una ventana temporal,
    eligiendo el par cuyo punto medio esté más cerca de t_ref.
    """
    idx_min = encontrar_minimos_locales(y)
    t_min = t[idx_min]

    mascara = (t_min >= ventana[0]) & (t_min <= ventana[1])
    mins = idx_min[mascara]

    if len(mins) < 2:
        raise ValueError(f"No hay suficientes mínimos en la ventana {ventana}")

    midpoints = 0.5 * (t[mins[:-1]] + t[mins[1:]])
    k = np.argmin(np.abs(midpoints - t_ref))

    i1 = mins[k]
    i2 = mins[k + 1]
    return i1, i2

def dibujar_periodo_desde_minimos(ax, t, y, i1, i2, etiqueta,
                                  margen_flecha=0.035, margen_texto=0.035):
    """
    Dibuja una doble flecha horizontal desde un mínimo hasta el mínimo consecutivo,
    cerca de la ondulación, y coloca la etiqueta debajo.
    """
    y_arrow = min(y[i1], y[i2]) - margen_flecha
    y_text = y_arrow - margen_texto

    ax.annotate(
        "",
        xy=(t[i1], y_arrow),
        xytext=(t[i2], y_arrow),
        arrowprops=dict(arrowstyle="<->", lw=2, color="black")
    )

    ax.text(
        0.5 * (t[i1] + t[i2]),
        y_text,
        etiqueta,
        ha="center",
        va="top",
        fontsize=12
    )

# ============================================================
# Mallas temporales
# ============================================================
t_comp = np.linspace(-0.8, 0.8, 2500)     # panel de componentes individuales
t_camp = np.linspace(-2.0, 2.0, 5000)     # campo + envolvente
t_int  = np.linspace(-16.5, 16.5, 7000)   # intensidad en ventana grande

# ============================================================
# Cálculos: fase constante
# ============================================================
E_const_comp = campo_real(t_comp, omega, A, phi_const)
E_const      = campo_real(t_camp, omega, A, phi_const)
Env_const    = envolvente(t_camp, omega, A, phi_const)
I_const      = intensidad(t_int, omega, A, phi_const)

E_const_n   = E_const / Env_const.max()
Env_const_n = Env_const / Env_const.max()
I_const_n   = I_const / I_const.max()

# ============================================================
# Cálculos: fase cuadrática
# ============================================================
E_quad_comp = campo_real(t_comp, omega, A, phi_quad)
E_quad      = campo_real(t_camp, omega, A, phi_quad)
Env_quad    = envolvente(t_camp, omega, A, phi_quad)
I_quad      = intensidad(t_int, omega, A, phi_quad)

E_quad_n   = E_quad / Env_quad.max()
Env_quad_n = Env_quad / Env_quad.max()
I_quad_n   = I_quad / I_quad.max()

# ============================================================
# Estilo global
# ============================================================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 17,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12
})

fig, ax = plt.subplots(5, 2, figsize=(14, 17))
fig.suptitle("Síntesis temporal por suma discreta de frecuencias", fontsize=24, y=0.985)

# ============================================================
# Columna izquierda: fase espectral constante
# ============================================================

# 1) Amplitudes
markerline, stemlines, baseline = ax[0, 0].stem(
    omega, A, linefmt='C0-', markerfmt='C0o', basefmt=' '
)
plt.setp(stemlines, linewidth=1.5)
plt.setp(markerline, markersize=6)
ax[0, 0].set_title("Fase espectral constante")
ax[0, 0].set_ylabel(r"$A_n$")
ax[0, 0].set_xlabel(r"$\omega_n$")
ax[0, 0].set_xlim(18.5, 51.5)
ax[0, 0].set_ylim(-0.05, 1.05)
ax[0, 0].grid(True, alpha=0.3)

# 2) Fases
ax[1, 0].plot(omega, phi_const, 'o-', lw=1.8, ms=6)
ax[1, 0].set_ylabel(r"$\phi_n$")
ax[1, 0].set_xlabel(r"$\omega_n$")
ax[1, 0].set_xlim(18.5, 51.5)
ax[1, 0].set_ylim(-0.055, 0.055)
ax[1, 0].grid(True, alpha=0.3)

# 3) Componentes individuales
for n in range(N):
    ax[2, 0].plot(t_comp, A[n] * np.cos(omega[n] * t_comp + phi_const[n]), lw=1.0)
ax[2, 0].set_ylabel(r"$A_n \cos(\omega_n t + \phi_n)$")
ax[2, 0].set_xlabel(r"$t$")
ax[2, 0].set_xlim(-0.88, 0.88)
ax[2, 0].set_ylim(-1.1, 1.1)
ax[2, 0].grid(True, alpha=0.3)

# 4) Campo + envolvente
ax[3, 0].plot(t_camp, E_const_n, lw=1.3, label="Campo")
ax[3, 0].plot(t_camp, Env_const_n, lw=2.0, label="Envolvente")
ax[3, 0].set_ylabel("Amplitud normalizada")
ax[3, 0].set_xlabel(r"$t$")
ax[3, 0].set_xlim(-2.2, 2.2)
ax[3, 0].set_ylim(-0.95, 1.1)
ax[3, 0].legend(loc="upper right")
ax[3, 0].grid(True, alpha=0.3)

# 5) Intensidad
ax[4, 0].plot(t_int, I_const_n, lw=2.0)
ax[4, 0].set_ylabel("Intensidad normalizada")
ax[4, 0].set_xlabel(r"$t$")
ax[4, 0].set_xlim(-16.5, 16.5)
ax[4, 0].set_ylim(-0.02, 1.05)
ax[4, 0].grid(True, alpha=0.3)

# ============================================================
# Columna derecha: fase espectral cuadrática
# ============================================================

# 1) Amplitudes
markerline, stemlines, baseline = ax[0, 1].stem(
    omega, A, linefmt='C0-', markerfmt='C0o', basefmt=' '
)
plt.setp(stemlines, linewidth=1.5)
plt.setp(markerline, markersize=6)
ax[0, 1].set_title("Fase espectral cuadrática")
ax[0, 1].set_ylabel(r"$A_n$")
ax[0, 1].set_xlabel(r"$\omega_n$")
ax[0, 1].set_xlim(18.5, 51.5)
ax[0, 1].set_ylim(-0.05, 1.05)
ax[0, 1].grid(True, alpha=0.3)

# 2) Fases
ax[1, 1].plot(omega, phi_quad, 'o-', lw=1.8, ms=6)
ax[1, 1].set_ylabel(r"$\phi_n$")
ax[1, 1].set_xlabel(r"$\omega_n$")
ax[1, 1].set_xlim(18.5, 51.5)
ax[1, 1].set_ylim(-0.5, 18.8)
ax[1, 1].grid(True, alpha=0.3)

# 3) Componentes individuales
for n in range(N):
    ax[2, 1].plot(t_comp, A[n] * np.cos(omega[n] * t_comp + phi_quad[n]), lw=1.0)
ax[2, 1].set_ylabel(r"$A_n \cos(\omega_n t + \phi_n)$")
ax[2, 1].set_xlabel(r"$t$")
ax[2, 1].set_xlim(-0.88, 0.88)
ax[2, 1].set_ylim(-1.1, 1.1)
ax[2, 1].grid(True, alpha=0.3)

# 4) Campo + envolvente
ax[3, 1].plot(t_camp, E_quad_n, lw=1.3, label="Campo")
ax[3, 1].plot(t_camp, Env_quad_n, lw=2.0, label="Envolvente")
ax[3, 1].set_ylabel("Amplitud normalizada")
ax[3, 1].set_xlabel(r"$t$")
ax[3, 1].set_xlim(-2.2, 2.2)
ax[3, 1].set_ylim(-1.05, 1.05)
ax[3, 1].legend(loc="upper right")
ax[3, 1].grid(True, alpha=0.3)

# ============================================================
# Elegimos un periodo local a la izquierda y otro a la derecha
# MEDIDOS DE MÍNIMO A MÍNIMO
# ============================================================
i1L, i2L = seleccionar_par_de_minimos(
    t_camp, E_quad_n, t_ref=-1.15, ventana=(-1.75, -0.35)
)
i1R, i2R = seleccionar_par_de_minimos(
    t_camp, E_quad_n, t_ref= 1.15, ventana=( 0.35,  1.75)
)

T_left = t_camp[i2L] - t_camp[i1L]
T_right = t_camp[i2R] - t_camp[i1R]

if T_left < T_right:
    etiqueta_left = r"$T_{\min}$"
    etiqueta_right = r"$T_{\max}$"
else:
    etiqueta_left = r"$T_{\max}$"
    etiqueta_right = r"$T_{\min}$"

# Dibujar las dos flechas cerca de la ondulación
dibujar_periodo_desde_minimos(
    ax[3, 1], t_camp, E_quad_n, i1L, i2L, etiqueta_left,
    margen_flecha=0.03, margen_texto=0.035
)

dibujar_periodo_desde_minimos(
    ax[3, 1], t_camp, E_quad_n, i1R, i2R, etiqueta_right,
    margen_flecha=0.03, margen_texto=0.035
)

# 5) Intensidad
ax[4, 1].plot(t_int, I_quad_n, lw=2.0)
ax[4, 1].set_ylabel("Intensidad normalizada")
ax[4, 1].set_xlabel(r"$t$")
ax[4, 1].set_xlim(-16.5, 16.5)
ax[4, 1].set_ylim(-0.02, 1.05)
ax[4, 1].grid(True, alpha=0.3)

# ============================================================
# Ajuste final
# ============================================================
plt.tight_layout(rect=[0, 0, 1, 0.975])
plt.show()

# Para guardar:
# fig.savefig("sintesis_temporal_por_suma_discreta_de_frecuencias_con_Tmin_Tmax.png",
#             dpi=300, bbox_inches="tight")