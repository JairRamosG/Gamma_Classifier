"""
Clasificador Gamma — implementación del algoritmo de clasificación basado en
memorias Alpha-Beta, operador gamma y código Johnson-Möbius modificado.

Extraído de ``clasificador_gamma_v2.ipynb``.
"""

from collections import defaultdict

import numpy as np

__all__ = [
    "operador_alpha",
    "operador_u",
    "longitud_binaria_minima",
    "codigo_johnson_mobius",
    "operador_gamma",
    "ClasificadorGamma",
]


# =============================================================================
# Bloque 1 — utils
# =============================================================================


def operador_alpha(x: int, y: int) -> int:
    """
    Operador α de las memorias Alpha-Beta:
        0  si x < y,
        1  si x = y,
        2  si x > y
    """
    if x < y:
        return 0
    elif x == y:
        return 1
    else:
        return 2


def operador_u(v: np.ndarray) -> int:
    """Operador u: cuenta cuántos 1 tiene el vector binario v."""
    return int(np.sum(v == 1))


def longitud_binaria_minima(r: int) -> int:
    """
    k(r): número mínimo de bits para representar r en código JM.
    """
    if r == 0:
        return 1
    return int(np.ceil(np.log2(r + 1))) + 1


def codigo_johnson_mobius(r: int, L: int) -> np.ndarray:
    """
    Código Johnson-Möbius Modificado E(r, L).
    Vector binario de longitud L con los primeros r bits en 1.
    """
    c = np.zeros(L, dtype=int)
    if r > 0:
        c[: min(r, L)] = 1
    return c


def operador_gamma(xj: np.ndarray, yj: np.ndarray, lam: int, nmax: int) -> int:
    """
    Parámetros
    ----------
    xj   : vector JM de la componente j del patrón fundamental (longitud em(j))
    yj   : vector JM de la componente j del patrón a clasificar (longitud em(j))
    lam  : tolerancia λ
    nmax : dimensión efectiva para la fórmula (n - u) <= λ
        - Si y[j] <= em(j): nmax = em(j)   (usa todos los bits del código)
        - Si y[j] >  em(j): nmax = y[j]    (valor original antes de truncar)

    Algoritmo:
        1. Operar sobre los primeros min(nmax, len(xj)) bits
        2. alpha(xj_i, yj_i) mod 2 para cada bit i
        3. u = contar 1s
        4.  (nmax - u) <= λ  →  1  (similares)
            (nmax - u) >  λ  →  0  (distintos)
    """
    n_op = min(nmax, len(xj))
    amod2 = np.array([operador_alpha(xj[i], yj[i]) % 2 for i in range(n_op)])
    u = operador_u(amod2)
    return 1 if (nmax - u) <= lam else 0


# =============================================================================
# Bloque 2 — Clasificador Gamma
# =============================================================================


class ClasificadorGamma:
    """
    Clasificador Gamma basado en memorias Alpha-Beta.

    Parámetros
    ----------
    factor_escala : float, default=1
        Factor aplicado a los patrones antes de redondear a entero.
    """

    def __init__(self, factor_escala: float = 1):
        self.factor_escala = factor_escala

    # ------------------------------------------------------------------
    # Preprocesamiento interno
    # ------------------------------------------------------------------

    def _escalar(self, X: np.ndarray) -> np.ndarray:
        """Aplica factor_escala y redondea a entero."""
        return np.round(np.array(X, dtype=float) * self.factor_escala).astype(int)

    def _desplazar(self, X: np.ndarray) -> np.ndarray:
        """
        Desplaza X por columna para eliminar negativos.
        Resta el mínimo negativo de cada columna.
        El offset se aprende en fit() y se reutiliza en predict().
        """
        return X - self._offset

    def _codificar_patron(self, patron: np.ndarray) -> list:
        """Codifica un patrón como lista de vectores JM, uno por componente."""
        return [
            codigo_johnson_mobius(int(patron[j]), self.em_[j])
            for j in range(self._n_attr)
        ]

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fase de entrenamiento.

        Paso 1: Transformar y escalar los patrones.
                em(j) = máximo de la componente j en el conjunto fundamental.
                Codificar cada componente con código JM de longitud em(j).
        Paso 2: δ = mínimo de todos los em(j).

        Parámetros
        ----------
        X : array (p, n)  — patrones de entrenamiento
        y : array (p,)    — etiqueta de clase de cada patrón
        """
        X = self._escalar(X)
        y = np.array(y)

        self.clases_ = sorted(np.unique(y).tolist())
        self._n_attr = X.shape[1]

        # Offset: restar mínimo negativo por columna (0 si la columna es positiva)
        self._offset = np.minimum(X.min(axis=0), 0)
        X = self._desplazar(X)

        # Paso 1 — em(j): máximo de cada columna tras el desplazamiento
        self.em_ = {j: int(X[:, j].max()) for j in range(self._n_attr)}

        # Paso 1 — codificar patrones y agrupar por clase
        # X_cod_[clase][ω][j] = vector JM de la componente j del patrón ω
        self.X_cod_ = defaultdict(list)
        for patron, clase in zip(X, y):
            self.X_cod_[clase].append(self._codificar_patron(patron))

        # Paso 2 — δ = min(em(j))
        self.delta_ = min(self.em_.values())

        return self

    # ------------------------------------------------------------------
    # Clasificación
    # ------------------------------------------------------------------

    def predict_one(self, patron: np.ndarray, verbose: bool = False) -> object:
        """
        Clasifica un único patrón.

        Paso 3 : Escalar y desplazar y con el mismo offset aprendido.
                Si y[j] > em(j): truncar a em(j) y guardar valor original como nmax.
        Paso 4 : Reindexar por clase (implícito en X_cod_).
        Paso 5 : λ = 0.
        Paso 6 : g(x^{iω}_j, y_j, λ) con nmax correspondiente.
        Paso 7 : ci = Σ_ω Σ_j g / ki
        Paso 8 : Sin máximo único y λ < δ → λ += 1 y repetir.
        Paso 9 : Máximo único → asignar esa clase.
        Paso 10: Empate en λ=δ → clase de menor índice entre empatadas.
        """
        patron = self._escalar(np.array([patron]))[0] - self._offset

        # Paso 3 — codificar y; calcular nmax por componente
        y_cod = []
        nmax_y = []
        for j in range(self._n_attr):
            val_orig = int(patron[j])
            val_trunc = min(val_orig, self.em_[j])
            y_cod.append(codigo_johnson_mobius(val_trunc, self.em_[j]))
            nmax_y.append(val_orig if val_orig > self.em_[j] else self.em_[j])

        if verbose:
            print(f"\n  y escalado    :\t {patron}")
            print(f"  nmax          :\t {nmax_y}")

        sumas = {}
        for lam in range(self.delta_ + 1):

            # Pasos 6-7 — sumas ponderadas
            for clase in self.clases_:
                patrones_cod = self.X_cod_[clase]
                ki = len(patrones_cod)
                total = sum(
                    operador_gamma(patrones_cod[w][j], y_cod[j], lam, nmax_y[j])
                    for w in range(ki)
                    for j in range(self._n_attr)
                )
                sumas[clase] = total / ki

            if verbose:
                vals = "  ".join(f"c({c})={v:.4f}" for c, v in sumas.items())
                print(f"  λ={lam} → {vals}")

            # Paso 8 — ¿máximo único?
            max_val = max(sumas.values())
            ganadores = [c for c, v in sumas.items() if v == max_val]

            if len(ganadores) == 1:  # Paso 9
                if verbose:
                    print(f"  → Clase asignada: {ganadores[0]}")
                return ganadores[0]

        # Paso 10 — empate persistente: menor índice (clases_ ya está ordenada)
        clase_final = next(c for c in self.clases_ if c in ganadores)
        if verbose:
            print(f"  → Empate persistente. Clase por menor índice: {clase_final}")
        return clase_final

    def predict(self, X: np.ndarray, verbose: bool = False) -> list:
        """Clasifica múltiples patrones. X debe ser array 2D (q, n)."""
        return [self.predict_one(p, verbose) for p in X]

    def resumen(self):
        """Imprime los parámetros aprendidos."""
        print("─" * 52)
        print("  Clasificador Gamma — parámetros aprendidos")
        print("─" * 52)
        print(f"  Clases        : {self.clases_}")
        print(f"  Atributos (n) : {self._n_attr}")
        print(f"  Offset        : {self._offset}")
        print(f"  em(j)         : {self.em_}")
        print(f"  δ max         : {self.delta_}")
        print("Patrones codificados:")
        for clase in self.clases_:
            print(f" Clase {clase}")
            for w, patron in enumerate(self.X_cod_[clase]):
                print(f"     -x{w + 1}")
                for j, vec in enumerate(patron):
                    print(f"         {j + 1} : ({vec[::-1]})")

        for c in self.clases_:
            print(f"  |C_{c}|        : {len(self.X_cod_[c])} patrones")
        print("─" * 52)
