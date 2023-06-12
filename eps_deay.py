import matplotlib.pyplot as plt
import numpy as np

epsilon_start = 1.0
num_training_steps = 2_056
# decay_factor = 0.999
epsilon_end = 0.1
decay_factor = (epsilon_end/epsilon_start) ** (1/num_training_steps)
epsilon = epsilon_start
print(decay_factor)
epsilons = [epsilon := epsilon * decay_factor for _ in range(num_training_steps)]
epsilons.insert(0, epsilon_start)
plt.Figure()
plt.plot(np.arange(num_training_steps+1), epsilons)
plt.show()