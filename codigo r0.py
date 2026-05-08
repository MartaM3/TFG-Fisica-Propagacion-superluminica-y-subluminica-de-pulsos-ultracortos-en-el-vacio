import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0


rho = np.linspace(0, 12, 2000)

# Intensidad normalizada
I = j0(rho)**2

# Primer cero de J0
rho0 = 2.405

fig, ax = plt.subplots(figsize=(7, 4))

# Curva principal
ax.plot(rho, I, lw=2, color="C0")


ax.axvline(rho0, ls="--", lw=1.4, color="black")

# Flecha  para marcar r0
y_arrow = 0.10
ax.annotate(
    "",
    xy=(rho0, y_arrow),
    xytext=(0, y_arrow),
    arrowprops=dict(arrowstyle="<->", lw=1.6, color="black")
)


ax.text(
    rho0/2,
    y_arrow + 0.05,
    r"$r_0=\dfrac{2.405}{k_r}$",
    ha="center",
    va="bottom",
    fontsize=13
)


ax.text(
    1.0,
    0.82,
    "Lóbulo central",
    ha="center",
    va="center",
    fontsize=12
)


ax.text(
    rho0 + 0.15,
    0.92,
    r"Primer cero de $J_0$",
    ha="left",
    va="center",
    fontsize=12
)

# Ejes
ax.set_xlabel(r"$k_r r$")
ax.set_ylabel("Intensidad normalizada")
ax.set_xlim(0, 12)
ax.set_ylim(-0.02, 1.05)

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("bessel_r0_profile.pdf", bbox_inches="tight")
plt.savefig("bessel_r0_profile.png", dpi=300, bbox_inches="tight")
plt.show()