"""
Ejemplos y pruebas de humo del Clasificador Gamma.

Este archivo SÍ está pensado para ejecutarse directamente:

    python3 ejemplos.py

También puedes importar cada ejemplo por separado:

    from ejemplos import ejemplo_4_8
    ejemplo_4_8()

Los dos primeros casos reproducen los ejemplos 4.8 y 4.15 de la tesis y
verifican el resultado esperado con un assert, así que sirven como prueba
de regresión rápida del algoritmo.
"""

import numpy as np

from clasificador_gamma import ClasificadorGamma

__all__ = ["ejemplo_4_8", "ejemplo_4_15", "ejemplo_iris", "main"]


def _titulo(texto: str) -> None:
    print("\n" + "=" * 52)
    print(f"  {texto}")
    print("=" * 52)


# =============================================================================
# Ejemplo 4.8 de la tesis — 4 patrones, 2 atributos, 2 clases
# =============================================================================


def ejemplo_4_8() -> list:
    """
    x¹ = [2, 8]   x² = [1, 9]   → C1
    x³ = [6, 3]   x⁴ = [7, 4]   → C2

    Se clasifican y1 = [6, 2] e y2 = [3, 9].
    Resultado esperado: y1 → C2, y2 → C1.
    """
    _titulo("Ejemplo 4.8 de la tesis")

    X_train = np.array(
        [
            [2, 8],  # x1 → C1
            [1, 9],  # x2 → C1
            [6, 3],  # x3 → C2
            [7, 4],  # x4 → C2
        ]
    )
    y_train = np.array(["C1", "C1", "C2", "C2"])

    X_test = np.array(
        [
            [6, 2],  # y1
            [3, 9],  # y2
        ]
    )

    clf = ClasificadorGamma().fit(X_train, y_train)
    clf.resumen()

    preds = clf.predict(X_test, verbose=True)
    print(f"\n  Predicciones: {preds}")

    assert preds == ["C2", "C1"], f"4.8 falló: se esperaba ['C2', 'C1'], se obtuvo {preds}"
    print("  ✓ Ejemplo 4.8 correcto")
    return preds


# =============================================================================
# Ejemplo 4.15 de la tesis — 5 patrones, 4 atributos, 2 clases
# =============================================================================


def ejemplo_4_15() -> list:
    """
    x¹ = [-8, 6, 4, 2]     x² = [-11, 10, 0, -1]   x⁴ = [-7, 7, 3, 0]  → C1
    x³ = [3, 15, 15, 10]   x⁵ = [5, 18, 10, 5]                        → C2

    Se clasifica y¹ = [0, 23, 7, 7].
    Resultado esperado: y¹ → C2.
    """
    _titulo("Ejemplo 4.15 de la tesis")

    X_train = np.array(
        [
            [-8, 6, 4, 2],  # x1 → C1
            [-11, 10, 0, -1],  # x2 → C1
            [3, 15, 15, 10],  # x3 → C2
            [-7, 7, 3, 0],  # x4 → C1
            [5, 18, 10, 5],  # x5 → C2
        ]
    )
    y_train = np.array(["C1", "C1", "C2", "C1", "C2"])

    X_test = np.array([[0, 23, 7, 7]])  # y1

    clf = ClasificadorGamma().fit(X_train, y_train)
    clf.resumen()

    preds = clf.predict(X_test, verbose=True)
    print(f"\n  Predicciones: {preds}")

    assert preds == ["C2"], f"4.15 falló: se esperaba ['C2'], se obtuvo {preds}"
    print("  ✓ Ejemplo 4.15 correcto")
    return preds


# =============================================================================
# Dataset Iris — 150 patrones, 4 atributos, 3 clases
# =============================================================================


def ejemplo_iris() -> float:
    """
    Clasifica el dataset Iris con una partición 20/80.

    Ojo: se pasa X_test (las características) a predict, NO y_test (las
    etiquetas). Pasar y_test produce resultados sin sentido porque el
    clasificador recibe un escalar y lo difunde sobre los atributos.

    Devuelve la accuracy obtenida.
    """
    try:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("\n  [omitido] Ejemplo Iris requiere scikit-learn.")
        print("            Instálalo con: pip install scikit-learn")
        return float("nan")

    _titulo("Dataset Iris")

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.8,
        random_state=0,
    )

    print(f"  Train: {X_train.shape[0]} patrones | Test: {X_test.shape[0]} patrones")

    clf = ClasificadorGamma().fit(X_train, y_train)
    clf.resumen()

    preds = clf.predict(X_test)
    accuracy = float(np.mean(np.array(preds) == y_test))

    print(f"\n  Primeras 20 predicciones: {preds[:20]}")
    print(f"  Primeras 20 etiquetas   : {y_test[:20].tolist()}")
    print(f"  Accuracy: {accuracy:.2%}")

    assert accuracy > 0.90, f"Iris: accuracy demasiado baja ({accuracy:.2%})"
    print("  ✓ Iris dentro de lo esperado")
    return accuracy


# =============================================================================
# Punto de entrada
# =============================================================================


def main() -> None:
    ejemplo_4_8()
    ejemplo_4_15()
    ejemplo_iris()
    print("\n" + "=" * 52)
    print("  Todos los ejemplos pasaron correctamente")
    print("=" * 52)


if __name__ == "__main__":
    main()
