import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.patches import Polygon

st.set_page_config(layout="centered")
st.title("📈 Calculadora de Programación Lineal - Método Gráfico")

st.markdown("""
Esta aplicación resuelve *5 casos de ejemplo* + tu **Personalizado**, todos con dos variables (x₁, x₂) y método gráfico.
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

# --- Selector de caso ---
selected = st.selectbox("🔍 Elige un caso de ejemplo o Personalizado", list(cases.keys()))

# --- Parámetros según selección ---
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

# --- Botón de resolución ---
if st.button("📊 Resolver y Graficar"):
    try:
        # Parsear
        constraints = []
        for line in input_text.strip().split("\n"):
            a1, a2, sign, b = line.split()
            a1, a2, b = float(a1), float(a2), float(b)
            if sign not in ("<=", ">="):
                raise ValueError("Signo inválido. Usa '<=' o '>='.")
            constraints.append((a1, a2, b, sign))
        # No negatividad
        constraints += [(1, 0, 0, ">="), (0, 1, 0, ">=")]

        # Preparar malla
        lim = max(max(c[2] for c in constraints)*1.2, 50)
        x = np.linspace(0, lim, 500)
        fig, ax = plt.subplots(figsize=(8,6))
        lines = []

        # Trazar rectas
        for (a1, a2, b, sign) in constraints:
            if abs(a2) > 1e-6:
                y = (b - a1*x)/a2
                lines.append(((a1,a2,b,sign), x, y))
                ax.plot(x, y, label=f"{a1}·x₁ {sign} {b}")
            else:
                xv = b/a1
                yy = np.linspace(0, lim, 500)
                lines.append(((a1,a2,b,sign), np.full_like(yy, xv), yy))
                ax.plot(np.full_like(yy, xv), yy, label=f"{a1}·x₁ {sign} {b}")

        # Intersecciones
        feasible = []
        for (c1,_,_), (c2,_,_) in combinations(lines, 2):
            A = np.array([[c1[0],c1[1]],[c2[0],c2[1]]])
            bvec = np.array([c1[2],c2[2]])
            try:
                sol = np.linalg.solve(A,bvec)
            except np.linalg.LinAlgError:
                continue
            if np.all(sol>=-1e-6):
                ok = True
                for (a1,a2,bi,sg) in constraints:
                    val = a1*sol[0] + a2*sol[1]
                    if (sg=="<=" and val>bi+1e-6) or (sg==">=" and val<bi-1e-6):
                        ok=False; break
                if ok:
                    feasible.append((sol[0],sol[1]))

        feasible = list(set(feasible))
        if not feasible:
            st.error("❌ No hay región factible.")
            st.pyplot(fig)
            st.stop()

        # Evaluar Z
        zvals = [obj1*x0 + obj2*x1 for x0,x1 in feasible]
        idx = np.argmax(zvals) if maximize else np.argmin(zvals)
        xp, yp, zp = *feasible[idx], zvals[idx]

        # Región factible
        if len(feasible) >= 3:
            poly = Polygon(feasible, color="lightgray", alpha=0.5)
            ax.add_patch(poly)

        # Punto óptimo
        ax.plot(xp, yp, "ro")
        ax.annotate(f"Pto óptimo\n({xp:.1f},{yp:.1f})\nZ={zp:.1f}",
                    xy=(xp,yp), xytext=(xp+1,yp+1),
                    arrowprops=dict(arrowstyle="->"))

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.set_title(f"{'Maximizar' if maximize else 'Minimizar'} función objetivo")
        ax.grid(True)
        ax.legend(loc="best")

        st.pyplot(fig)
        st.success(f"✅ Solución: x₁={xp:.2f}, x₂={yp:.2f}")
        st.info(f"{'Máximo' if maximize else 'Mínimo'} Z = {zp:.2f}")

    except Exception as e:
        st.error(f"Error al resolver: {e}")
