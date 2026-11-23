import numpy as np
import matplotlib.pyplot as plt


def taylor_ln(x, n):
    """Polinomio de Taylor de grado n de ln(x) alrededor de x0 = 1."""
    x = np.asarray(x, dtype=float)
    suma = np.zeros_like(x, dtype=float)
    for k in range(1, n + 1):
        suma = suma + ((-1) ** (k + 1)) * ((x - 1) ** k) / k
    return float(suma) if suma.ndim == 0 else suma


def comprobar(x, tolerancia=1e-6, max_n=200):
    valor_real = np.log(x)
    n = 0
    error = np.abs(valor_real - taylor_ln(x, n))
    while error >= tolerancia and n < max_n:
        n += 1
        error = np.abs(valor_real - taylor_ln(x, n))
    return n, taylor_ln(x, n), error


def graficar_aproximaciones(
    grados=(1, 2, 3, 6),
    x_min=0.1,
    x_max=3.0,
    archivo="aproximacion_taylor_ln.png",
    mostrar=True,
):
    """Grafica ln(x) y sus polinomios de Taylor alrededor de x0 = 1, y guarda la imagen."""
    x_vals = np.linspace(x_min, x_max, 400)
    y_exacta = np.log(x_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, y_exacta, label=r"$\ln(x)$", color="black", linewidth=2.2)

    for grado in grados:
        ax.plot(
            x_vals,
            taylor_ln(x_vals, grado),
            label=rf"$P_{grado}(x)$",
            linestyle="--",
        )

    ax.scatter([1.0], [0.0], color="red", zorder=5, label=r"Centro $(1, 0)$")

    ax.set_ylim(-3, 2)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"Aproximación de Taylor de $\ln(x)$ alrededor de $x_0 = 1$")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(archivo, dpi=300, bbox_inches="tight")

    if mostrar:
        plt.show()
    plt.close(fig)


def graficar_error(
    n_max=30,
    archivo="error_taylor_ln.png",
    mostrar=True,
):
    """Grafica el error absoluto en escala logarítmica para x=1.5 y x=2.5, y guarda la imagen."""
    ns = np.arange(1, n_max + 1)
    err_15 = np.array([np.abs(np.log(1.5) - taylor_ln(1.5, int(n))) for n in ns])
    err_25 = np.array([np.abs(np.log(2.5) - taylor_ln(2.5, int(n))) for n in ns])

    cota_15 = (0.5 ** (ns + 1)) / (ns + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(ns, err_15, marker="o", markersize=4, label=r"$|\ln(1.5) - P_n(1.5)|$")
    ax.semilogy(
        ns,
        cota_15,
        linestyle="--",
        color="tab:green",
        label=r"Cota en $x=1.5$: $\frac{0.5^{n+1}}{n+1}$",
    )
    ax.semilogy(ns, err_25, marker="s", markersize=4, label=r"$|\ln(2.5) - P_n(2.5)|$")

    ax.set_xlabel(r"Grado del polinomio $n$")
    ax.set_ylabel(r"Error absoluto (escala log)")
    ax.set_title(rf"Error de aproximación de $\ln(x)$ para $n = 1, \dots, {n_max}$")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(archivo, dpi=300, bbox_inches="tight")

    if mostrar:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    p3_15 = taylor_ln(1.5, 3)
    print(f"P3(1.5) = {p3_15:.6f}")

    p10_15 = taylor_ln(1.5, 10)
    print(f"P10(1.5) = {p10_15:.6f}")

    p10_25 = taylor_ln(2.5, 10)
    print(f"P10(2.5) = {p10_25:.6f}")

    n_tol, aprox_tol, err_tol = comprobar(1.5, tolerancia=1e-6)
    print(
        f"Para x=1.5 y tol=1e-6 se requiere n={n_tol}: "
        f"P_{n_tol}(1.5)={aprox_tol:.8f}, error={err_tol:.2e}"
    )

    graficar_aproximaciones(archivo="aproximacion_taylor_ln.png", mostrar=True)
    graficar_error(archivo="error_taylor_ln.png", mostrar=True)