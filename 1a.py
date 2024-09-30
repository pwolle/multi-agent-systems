# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
## Question 1 a)
SIZE = 1000

# number of agetns
n = 1
# speed of agent
Rv = 25

# task capacity
Tc = 1

# task radius
Tr = 50

agent_x = np.random.randint((SIZE, SIZE)).astype(float)
agent_xs = [agent_x]

agent_v = np.random.normal(0, 1, 2)
agent_v = agent_v / np.linalg.norm(agent_v)

task_x = np.random.randint((SIZE, SIZE))
task_xs = [task_x]
task_ts = []

dt = 0.1
t = 0
t_max = 100_000

while t < t_max:
    t += dt
    agent_x = agent_x + agent_v * Rv * dt

    agent_v = np.where(agent_x < 0, -agent_v, agent_v)
    agent_v = np.where(agent_x > SIZE, -agent_v, agent_v)

    agent_v_prime = np.random.normal(0, 1, 2)
    agent_v_prime = agent_v_prime / np.linalg.norm(agent_v_prime)

    agent_v = agent_v + agent_v_prime * dt
    agent_v = agent_v / np.linalg.norm(agent_v)

    agent_xs.append(agent_x)

    # spawn new task
    if np.linalg.norm(agent_x - task_x) <= Tr:
        task_x = np.random.randint((SIZE, SIZE))

        task_xs.append(task_x)
        task_ts.append(t)


agent_xs = np.stack(agent_xs, axis=0)
task_xs = np.stack(task_xs, axis=0)

# %%
fig, ax = plt.subplots(figsize=(4, 4))

nplot = 10000

ax.plot(
    agent_xs[:nplot, 0],
    agent_xs[:nplot, 1],
    color="tab:blue",
    label=f"First {nplot} steps",
)

nplot = 100000

ax.plot(
    agent_xs[:nplot, 0],
    agent_xs[:nplot, 1],
    alpha=0.4,
    color="tab:blue",
    label=f"First {nplot} steps",
)

ax.set_xlabel("x")
ax.set_ylabel("y")

ax.set_xlim(0, SIZE)
ax.set_ylim(0, SIZE)

plt.title("Agent path")

plt.legend(
    frameon=True,
    loc="lower right",
)

plt.savefig(
    "plots/1a_path.pdf",
    transparent=True,
    bbox_inches="tight",
)

# %%
fig, ax = plt.subplots(figsize=(4, 4))

ax.plot(
    [0] + task_ts,
    range(len(task_ts) + 1),
    # np.linspace(0, t_max, len(task_ts) + 1),
    markersize=1,
)

sns.despine()

ax.set_xlabel("Time")
ax.set_ylabel("Number of tasks completed")

ax.set_xlim(0, None)
ax.set_ylim(0, None)

plt.title("Number of tasks completed over time")

plt.savefig(
    "plots/1a_perf.pdf",
    transparent=True,
    bbox_inches="tight",
)
len(task_ts) + 1

# %%
fig, ax = plt.subplots(figsize=(4, 4))

diff = np.diff([0] + task_ts)
plt.hist(
    diff,
    bins=32,
)

mean = np.mean(diff)
ax.axvline(
    float(mean),
    color="tab:orange",
    label=f"Mean: {mean:.2f}",
)

plt.legend(frameon=False)

ax.set_xlabel("Time between tasks")
ax.set_ylabel("Frequency")
plt.title("Time between tasks")

sns.despine()

plt.savefig(
    "plots/1a_hist.pdf",
    transparent=True,
    bbox_inches="tight",
)

# %%
