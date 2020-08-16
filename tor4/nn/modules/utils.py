import typing as t


def _pair(value: t.Union[int, t.Tuple[int, int]]) -> t.Tuple[int, int]:
    if isinstance(value, int):
        value = (value, value)

    return value
