import numpy as np


def gyroid(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)


def schwarz_p(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return np.cos(X) + np.cos(Y) + np.cos(Z)


def diamond(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return (
        np.sin(X) * np.sin(Y) * np.sin(Z)
        + np.sin(X) * np.cos(Y) * np.cos(Z)
        + np.cos(X) * np.sin(Y) * np.cos(Z)
        + np.cos(X) * np.cos(Y) * np.sin(Z)
    )


def neovius(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return 3 * (np.cos(X) + np.cos(Y) + np.cos(Z)) + 4 * np.cos(X) * np.cos(Y) * np.cos(
        Z
    )


def iwp(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return 2 * (
        np.cos(X) * np.cos(Y) + np.cos(Y) * np.cos(Z) + np.cos(Z) * np.cos(X)
    ) - (np.cos(2 * X) + np.cos(2 * Y) + np.cos(2 * Z))


def lidinoid(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return (
        0.5
        * (
            np.sin(2 * X) * np.cos(Y) * np.sin(Z)
            + np.sin(2 * Y) * np.cos(Z) * np.sin(X)
            + np.sin(2 * Z) * np.cos(X) * np.sin(Y)
        )
        - 0.5
        * (
            np.cos(2 * X) * np.cos(2 * Y)
            + np.cos(2 * Y) * np.cos(2 * Z)
            + np.cos(2 * Z) * np.cos(2 * X)
        )
        + 0.15
    )
