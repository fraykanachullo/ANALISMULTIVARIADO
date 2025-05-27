import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.patches import Polygon
from scipy.optimize import linprog

st.set_page_config(layout="centered")
st.title("📈 Calculadora de Programación Lineal")

st.markdown("""
Esta aplicación resuelve *5 casos de ejemplo* + tu **Personalizado**, todos con dos variables (x₁, x₂) y método gráfico.
También incluye el método simplex para resolver problemas generales.
""")

# --- Definición de los 5 casos de ejemplo ---
cases = {
    "Caso 1 (Min): C = 12·x₁ + 8·x₂": {
        "obj": (12.0, 8.0),
        "maximize": False,
        "constraints": [
            "2 2 >= 16",
            "4 1 >= 20"
        ]
    },
    "Caso 2 (Min): C = 80·x₁ + 100·x₂": {
        "obj": (80.0, 100.0),
        "maximize": False,
        "constraints": [
            "2 2 >= 80",
            "6 2 >= 120",
            "4 12 >= 240"
        ]
    },
    "Caso 3 (Min): P = 24·x₁ + 28·x₂": {
        "obj": (24.0, 28.0),
        "maximize": False,
        "constraints": [
            "5 4 <= 2000",
            "1 1 >= 300",
            "1 0 >= 80",
            "0 1 >= 100"
        ]
    },
    "Caso 4 (Max): U = 3·x₁ + x₂": {
        "obj": (3.0, 1.0),
        "maximize": True,
        "constraints": [
            "2 1 <= 8",
            "2 3 <= 12"
        ]
    },
    "Caso 5 (Max): G = 40·x₁ + 60·x₂": {
        "obj": (40.0, 60.0),
        "maximize": True,
        "constraints": [
            "1 1 <= 60",
            "1 3 <= 120"
        ]
    },
    "Personalizado": None
}

# --- Selector método ---
method = st.radio("Selecciona el método de solución:", ("Método Gráfico (2 variables)", "Método Simplex (general)"))

# --- Selector de caso ---
selected = st.selectbox("🔍 Elige un caso de ejemplo o Personalizado", list(cases.keys()))

if selected != "Personalizado":
    cfg = cases[selected]
    obj1, obj2 = cfg["obj"]
    maximize = cfg["maximize"]
    input_text = "\n".join(cfg["constraints"])
    st.info(f"📋 Cargando **{selected}**")
else:
    col1, col2 = st.columns(2)
    with col1:
        obj1 = st.number_input("Coeficiente de x₁ en la función objetivo", value=12.0)
    with col2:
        obj2 = st.number_input("Coeficiente de x₂ en la función objetivo", value=8.0)
    maximize = st.radio("Tipo de optimización:", ["Maximizar", "Minimizar"]) == "Maximizar"
    st.markdown("Escribe las restricciones en formato `a1 a2 signo b` (una por línea).")
    input_text = st.text_area("Restricciones", value="2 2 >= 16\n-4 1 >= 20", height=150)

if st.button("📊 Resolver"):
    try:
        # Parsear restricciones
        constraints = []
        for line in input_text.strip().split("\n"):
            a1, a2, sign, b = line.split()
            a1, a2, b = float(a1), float(a2), float(b)
            if sign not in ("<=", ">="):
                raise ValueError("Signo inválido. Usa '<=' o '>='.")
            constraints.append((a1, a2, b, sign))

        # Agregar no negatividad
        constraints += [(1, 0, 0, ">="), (0, 1, 0, ">=")]

        if method == "Método Gráfico (2 variables)":
            # Preparar gráfico solo si hay 2 variables
            lim = max(max(c[2] for c in constraints) * 1.2, 50)
            x = np.linspace(0, lim, 500)
            fig, ax = plt.subplots(figsize=(8, 6))
            lines = []

            for (a1, a2, b, sign) in constraints:
                if abs(a2) > 1e-6:
                    y = (b - a1 * x) / a2
                    lines.append(((a1, a2, b, sign), x, y))
                    ax.plot(x, y, label=f"{a1}·x₁ {sign} {b}")
                else:
                    xv = b / a1
                    yy = np.linspace(0, lim, 500)
                    lines.append(((a1, a2, b, sign), np.full_like(yy, xv), yy))
                    ax.plot(np.full_like(yy, xv), yy, label=f"{a1}·x₁ {sign} {b}")

            feasible = []
            for (c1, _, _), (c2, _, _) in combinations(lines, 2):
                A = np.array([[c1[0], c1[1]], [c2[0], c2[1]]])
                bvec = np.array([c1[2], c2[2]])
                try:
                    sol = np.linalg.solve(A, bvec)
                except np.linalg.LinAlgError:
                    continue
                if np.all(sol >= -1e-6):
                    ok = True
                    for (a1, a2, bi, sg) in constraints:
                        val = a1 * sol[0] + a2 * sol[1]
                        if (sg == "<=" and val > bi + 1e-6) or (sg == ">=" and val < bi - 1e-6):
                            ok = False
                            break
                    if ok:
                        feasible.append((sol[0], sol[1]))

            feasible = list(set(feasible))
            if not feasible:
                st.error("❌ No hay región factible.")
                st.pyplot(fig)
                st.stop()

            zvals = [obj1 * x0 + obj2 * x1 for x0, x1 in feasible]
            idx = np.argmax(zvals) if maximize else np.argmin(zvals)
            xp, yp, zp = *feasible[idx], zvals[idx]

            if len(feasible) >= 3:
                poly = Polygon(feasible, color="lightgray", alpha=0.5)
                ax.add_patch(poly)

            ax.plot(xp, yp, "ro")
            ax.annotate(f"Pto óptimo\n({xp:.1f},{yp:.1f})\nZ={zp:.1f}",
                        xy=(xp, yp), xytext=(xp + 1, yp + 1),
                        arrowprops=dict(arrowstyle="->"))

            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.set_title(f"{'Maximizar' if maximize else 'Minimizar'} función objetivo")
            ax.grid(True)
            ax.legend(loc="best")

            st.pyplot(fig)
            st.success(f"✅ Solución: x₁={xp:.2f}, x₂={yp:.2f}")
            st.info(f"{'Máximo' if maximize else 'Mínimo'} Z = {zp:.2f}")

        else:
            # Método Simplex general con scipy.optimize.linprog
            # Construir matrices para linprog
            c = np.array([obj1, obj2])
            if maximize:
                c = -c  # Para maximizar, minimizar el negativo

            A_ub = []
            b_ub = []

            A_lb = []
            b_lb = []

            for (a1, a2, b, sign) in constraints:
                if sign == "<=":
                    A_ub.append([a1, a2])
                    b_ub.append(b)
                elif sign == ">=":
                    # Convertir >= en <= multiplicando por -1
                    A_ub.append([-a1, -a2])
                    b_ub.append(-b)

            A_ub = np.array(A_ub) if A_ub else None
            b_ub = np.array(b_ub) if b_ub else None

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None))

            if res.success:
                x_opt = res.x
                z_opt = res.fun
                if maximize:
                    z_opt = -z_opt
                st.success(f"✅ Método Simplex: Solución óptima encontrada")
                st.write(f"x₁ = {x_opt[0]:.4f}")
                st.write(f"x₂ = {x_opt[1]:.4f}")
                st.write(f"Valor óptimo de Z = {z_opt:.4f}")
            else:
                st.error("❌ Método Simplex: No se encontró solución óptima.")
                st.write(res.message)

    except Exception as e:
        st.error(f"Error al resolver: {e}")
