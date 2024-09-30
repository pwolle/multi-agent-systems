# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

# %%
## Question 1 a)

SIZE = 1000

# number of agents
R = 5

# speed of agent
Rv = 25

# number of tasks
# T = 1

# task capacity
Tc = 1

# task radius
Tr = 50

dt = 0.1
t = 0
t_max = 1000

ts = np.linspace(0, t_max, int(t_max / dt))

fig, ax = plt.subplots(figsize=(4, 4))

ends = []
diffs = []


for T in [2, 10, 20]:

    # number of agetns
    # R = 1

    # agent_x = np.random.randint((R, SIZE, SIZE)).astype(float)
    agent_x = np.random.uniform(0, SIZE, (R, 2))
    agent_xs = [agent_x]

    agent_v = np.random.normal(0, 1, (R, 2))
    agent_v = agent_v / np.linalg.norm(agent_v)

    task_x = np.random.uniform(0, SIZE, (T, 2))
    task_s = [0]

    diff = []

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

        if cond.sum() > 0:
            diff.append(t)

        task_s.append(task_s[-1] + cond.sum())

        task_x_new = np.random.uniform(0, SIZE, (T, 2))
        # print(task_x_new.shape, task_x.shape, cond.shape)

        # print(cond.shape, task_x_new.shape, task_x.shape, t)
        task_x = np.where(cond[:, None], task_x_new, task_x)

    ends.append(task_s[-1])
    diffs.append(diff)
    plt.plot(ts, task_s[:-1], label=f"T={T}")


sns.despine()

ax.set_xlabel("Time")
ax.set_ylabel("Number of tasks completed")

ax.set_xlim(0, None)
ax.set_ylim(0, None)

plt.title("Varying number of tasks")

plt.legend(frameon=False, title=f"Tc={Tc}, R={R}")
plt.savefig("plots/1d.pdf", transparent=True, bbox_inches="tight")
# %%

fig, ax = plt.subplots(figsize=(4, 4))

diffs_mean = []
diffs_errs = []

for diff in diffs:
    diff = np.diff(diff)
    diffs_mean.append(np.mean((diff)))
    diffs_errs.append(np.std(diff) / np.sqrt(len(diff)))


plt.errorbar(
    [2, 10, 20],
    diffs_mean,
    yerr=diffs_errs,
    fmt="-",
    # marker="o",
    # ecolor="black",
    capsize=5,
)

sns.despine()

plt.xlabel("Number of tasks")
plt.ylabel("Time to complete the next task")

plt.title("Inter-task time with varying number of tasks")

plt.yscale("log")

plt.savefig("plots/1d_diff.pdf", transparent=True, bbox_inches="tight")
