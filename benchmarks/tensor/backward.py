import tracemalloc

import tor4


def main():
    tracemalloc.start()
    a = tor4.randn(1024, 1024, requires_grad=True, dtype=tor4.float32)
    b = tor4.randn(1024, 1024, dtype=tor4.float32)
    s = (a ** 2 + 2 * a * b + b ** 2).sum()

    s.backward(retain_graph=True)
    current, peak = tracemalloc.get_traced_memory()
    current = round(current / 1024 / 1024)
    peak = round(peak / 1024 / 1024)
    print(f"Current memory usage is {current}Mb; Peak was {peak}Mb")
    tracemalloc.stop()


def main2():
    tracemalloc.start()
    a = tor4.randn(1024, 1024, requires_grad=True, dtype=tor4.float32)
    b = tor4.randn(1024, 1024, dtype=tor4.float32)
    s = (a ** 2 + 2 * a * b + b ** 2).sum()

    s.backward(retain_graph=False)
    current, peak = tracemalloc.get_traced_memory()
    current = round(current / 1024 / 1024)
    peak = round(peak / 1024 / 1024)
    print(f"Current memory usage is {current}Mb; Peak was {peak}Mb")
    tracemalloc.stop()


if __name__ == "__main__":
    main()
    main2()
