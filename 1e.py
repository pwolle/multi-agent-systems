# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

# %%
## Question 1 a)

SIZE = 1000

# number of agents
R = 30

# speed of agent
Rv = 25

# number of tasks
T = 2

# task capacity
Tc = 3

# task radius
Tr = 50

Rd = 500

dt = 0.1
t = 0
t_max = 100000

ts = np.linspace(0, t_max, int(t_max / dt))

fig, ax = plt.subplots(figsize=(4, 4))

np.random.seed(42)

# for T in [2, 10, 20]:

# number of agetns
# R = 1

diffs = []
ends = []


for Rd in [100, 200, 300, 400, 600, 1000, 1400]:
    # for Rd in [100, 200, 400, 800, 1600]:
    # agent_x = np.random.randint((R, SIZE, SIZE)).astype(float)
    agent_x = np.random.uniform(0, SIZE, (R, 2))
    agent_xs = [agent_x]

    agent_v = np.random.normal(0, 1, (R, 2))
    agent_v = agent_v / np.linalg.norm(agent_v)

    task_x = np.random.uniform(0, SIZE, (T, 2))
    task_s = [0]

    diff = []

    for t in tqdm(ts):

        agent_v = np.where(agent_x < 0, -agent_v, agent_v)
        agent_v = np.where(agent_x > SIZE, -agent_v, agent_v)

        agent_v_p = np.random.normal(0, 1, 2)
        agent_v_p = agent_v_p / np.linalg.norm(agent_v_p, axis=-1, keepdims=True)

        agent_v = agent_v + agent_v_p * dt
        agent_v = agent_v / np.linalg.norm(agent_v, axis=-1, keepdims=True)

        # update tasks
        dists = np.linalg.norm(agent_x[..., None, :] - task_x[None, ...], axis=-1)
        cond = (dists <= Tr).sum(axis=0) >= Tc
        task_s.append(task_s[-1] + cond.sum())

        if cond.sum() > 0:
            diff.append(t)

        task_x_new = np.random.uniform(0, SIZE, (T, 2))
        task_x = np.where(cond[:, None], task_x_new, task_x)

        dists = np.linalg.norm(agent_x[..., None, :] - task_x[None, ...], axis=-1)
        call = (dists <= Tr).sum(axis=1) >= 1  # agent is near one or more tasks

        if np.any(call >= 1):
            dists = np.linalg.norm(
                agent_x[..., None, :] - agent_x[None, ...],
                axis=-1,
            )
            dists = dists + np.eye(R) * 1e8
            dists[~call] = 1e8
            nearest = np.argmin(dists, axis=1)

            direction = agent_x[nearest] - agent_x
            direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)

            cond = dists <= Rd  # which agent could reach the other agent
            cond = cond.sum(axis=1) >= 1  # at least one agent can reach the other agent
            cond = cond & ~call  # agent is not near any task

            agent_x = np.where(
                cond[:, None],
                agent_x + direction * Rv * dt,
                agent_x + agent_v * Rv * dt,
            )

        else:
            agent_x = agent_x + agent_v * Rv * dt

        agent_xs.append(agent_x)

    diffs.append(diff)
    ends.append(task_s[-1])
    plt.plot(ts, task_s[:-1], label=f"Rd={Rd}")


sns.despine()

ax.set_xlabel("Time")
ax.set_ylabel("Number of tasks completed")

ax.set_xlim(0, None)
ax.set_ylim(0, None)

plt.title("Varying number of tasks")

plt.legend(frameon=False, title=f"Tc={Tc}, R={R}")
plt.savefig("plots/1e.pdf", transparent=True, bbox_inches="tight")

# %%
fig, ax = plt.subplots(figsize=(4, 4))

plt.plot([100, 200, 300, 400, 600, 1000, 1400], ends, marker="o")

sns.despine()

ax.set_xlabel("Rd")
ax.set_ylabel("Number of tasks completed")

plt.title("Varying Rd")

plt.savefig("plots/1e_end.pdf", transparent=True, bbox_inches="tight")

# %%

fig, ax = plt.subplots(figsize=(4, 4))

diffs_mean = []
diffs_errs = []

for diff in diffs:
    diff = np.diff(diff)
    diffs_mean.append(np.mean((diff)))
    diffs_errs.append(np.std(diff) / np.sqrt(len(diff)))


plt.errorbar(
    # [100, 200, 400, 800, 1600],
    [100, 200, 300, 400, 600, 1000, 1400],
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

plt.savefig("plots/1e_diff.pdf", transparent=True, bbox_inches="tight")
