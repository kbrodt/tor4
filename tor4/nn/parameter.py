from ..tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data: Tensor) -> None:
        super().__init__(data=data.detach().cpu().numpy(), requires_grad=True)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"Parameter containing:\n{super().__repr__()}"
