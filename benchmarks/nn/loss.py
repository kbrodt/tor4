import sys

import tor4
import tor4.nn as nn


def main():
    method = sys.argv[1]

    a = nn.Parameter(tor4.randn(512, 512))
    b = tor4.randn(512, 512)
    if method == "cross_entropy":
        b = b.argmax(-1)

    for _ in range(25):
        loss = getattr(nn.functional, method)(a, b)
        loss.backward()


if __name__ == "__main__":
    main()
