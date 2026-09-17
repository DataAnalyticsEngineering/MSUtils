from dataclasses import dataclass

import numpy as np


def validate_order(order: str) -> str:
    if isinstance(order, bytes):
        order = order.decode("utf-8")
    if order not in {"xyz", "zyx"}:
        raise ValueError("Invalid order specified. Use 'xyz' or 'zyx'.")
    return order


def values_in_order(values, order: str) -> tuple:
    values = tuple(values)
    return values[::-1] if order == "zyx" else values


def image_in_order(image: np.ndarray, order: str) -> np.ndarray:
    return image.transpose(2, 1, 0) if order == "zyx" else image


@dataclass(frozen=True)
class GridSpec:
    """Description of a cell-centered three-dimensional grid in XYZ order."""

    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        lengths = tuple(self.lengths)

        if len(shape) != 3 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in shape
        ):
            raise ValueError("shape must contain three positive integers.")
        if len(lengths) != 3 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or value <= 0
            for value in lengths
        ):
            raise ValueError("lengths must contain three positive finite values.")
        object.__setattr__(self, "shape", tuple(int(value) for value in shape))
        object.__setattr__(self, "lengths", tuple(float(value) for value in lengths))

    @classmethod
    def from_h5_attributes(
        cls, shape: tuple[int, int, int], attributes, order: str
    ) -> "GridSpec":
        return cls(
            shape=shape,
            lengths=values_in_order(attributes.get("L", (1.0, 1.0, 1.0)), order),
        )

    def to_h5_attributes(self, order: str) -> dict[str, object]:
        order = validate_order(order)
        return {
            "permute_order": order,
            "L": values_in_order(self.lengths, order),
        }
