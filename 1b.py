# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

# %%
## Question 1 b)

fig, ax = plt.subplots(figsize=(4, 4))
ends = []


for R in [3, 5, 10, 20, 30]:
    SIZE = 1000

    # number of agetns
    # R = 1

    # speed of agent
    Rv = 25

    # number of tasks
    T = 1

    # task capacity
    Tc = 1

    # task radius
    Tr = 50

    # agent_x = np.random.randint((R, SIZE, SIZE)).astype(float)
    agent_x = np.random.uniform(0, SIZE, (R, 2))
    agent_xs = [agent_x]

    agent_v = np.random.normal(0, 1, (R, 2))
    agent_v = agent_v / np.linalg.norm(agent_v)

    task_x = np.random.uniform(0, SIZE, (T, 2))
    task_s = [0]

    dt = 0.1
    t = 0
    t_max = 10000

    ts = np.linspace(0, t_max, int(t_max / dt))

    for t in tqdm(ts):
        agent_x = agent_x + agent_v * Rv * dt

        agent_v = np.where(agent_x < 0, -agent_v, agent_v)
        agent_v = np.where(agent_x > SIZE, -agent_v, agent_v)

        agent_v_p = np.random.normal(0, 1, 2)
        agent_v_p = agent_v_p / np.linalg.norm(agent_v_p, axis=-1, keepdims=True)

        agent_v = agent_v + agent_v_p * dt
        agent_v = agent_v / np.linalg.norm(agent_v, axis=-1, keepdims=True)

        agent_xs.append(agent_x)

        dists = np.linalg.norm(agent_x[..., None, :] - task_x[None, ...], axis=-1)
        cond = (dists <= Tr).sum(axis=0) >= Tc

        task_s.append(task_s[-1] + cond.sum())

        task_x_new = np.random.uniform(0, SIZE, (T, 2))
        task_x = np.where(cond, task_x_new, task_x)

    plt.plot(ts, task_s[:-1], label=f"R={R}")
    ends.append(task_s[-1])


sns.despine()

ax.set_xlabel("Time")
ax.set_ylabel("Number of tasks completed")

ax.set_xlim(0, None)
ax.set_ylim(0, None)

plt.title("Varying number of agents")

plt.legend(frameon=False)
plt.savefig("plots/1b.pdf", transparent=True, bbox_inches="tight")

# %%
fig, ax = plt.subplots(figsize=(4, 4))

# for n, t in zip([3, 5, 10, 20, 30], ends):
#     ax.scatter(n, t, color="tab:blue")

plt.plot(
    [3, 5, 10, 20, 30],
    ends,
    color="tab:blue",
    marker="o",
)

ax.set_xlabel("Number of agents")
ax.set_ylabel("Number of tasks completed")

sns.despine()

plt.title("Varying number of agents")

plt.savefig("plots/1b_end.pdf", transparent=True, bbox_inches="tight")
