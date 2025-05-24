import math
import random
import numpy as np
import matplotlib.pyplot as plt


def get_samples():
    u_1 = random.random()
    u_2 = random.random()
    z_0 = math.sqrt(-2 * math.log(u_1)) * math.cos(2 * math.pi * u_2)
    z_1 = math.sqrt(-2 * math.log(u_2)) * math.cos(2 * math.pi * u_1)

    return (z_0, z_1)


if __name__ == "__main__":
    all_samples = []
    for i in range(1000):
        samples = get_samples()
        all_samples += samples

    all_samples = np.array(all_samples)

    plt.figure(figsize=(10, 6))
    plt.hist(all_samples, bins=25, color="blue")
    plt.title("Distribution of Box-Muller Generated Numbers")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.savefig("distribution.png")
