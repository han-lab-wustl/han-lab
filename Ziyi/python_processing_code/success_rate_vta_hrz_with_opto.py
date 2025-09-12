import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
npz_path = "success_rate_summary.npz"
data = np.load(npz_path, allow_pickle=True)

# Groups
opto_days_list = [32, 34, 35, 37, 39, 41,43,45,47,49]
control_days_list = [27, 30, 36, 38, 40, 42,44,46,48]

available = {int(k.split('_')[1]) for k in data.keys()}
opto_days = [d for d in opto_days_list if d in available]
control_days = [d for d in control_days_list if d in available and d not in opto_days]

def vals(days, key):
    return np.array([data[f"day_{d}"].item().get(key, np.nan) for d in days], float)

def mean_sem(a):
    a = a[~np.isnan(a)]
    n = len(a)
    mean = np.nan if n == 0 else a.mean()
    sem = np.nan if n <= 1 else a.std(ddof=1) / np.sqrt(n)
    return mean, sem, n

def pval(a, b):
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) > 1 and len(b) > 1:
        return stats.ttest_ind(a, b, equal_var=False).pvalue
    return np.nan

def stars(p):
    if np.isnan(p): return "NA"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

conds = [
    ("opto_ep", "Opto epoch"),
    ("nonopto_ep", "Non-opto epoch"),
    ("ep_after_opto", "Epoch after opto"),
]

# Colors for groups
c_opto = "C0"
c_ctrl = "C1"
alpha_pts = 0.45
rng = np.random.default_rng(7)

fig, ax = plt.subplots(figsize=(10, 5))
group_x = np.arange(len(conds))
dx = 0.18  # horizontal offset for the two groups within each condition

ymax = 0
for gi, (key, title) in enumerate(conds):
    a = vals(opto_days, key)
    b = vals(control_days, key)
    m1, s1, _ = mean_sem(a)
    m2, s2, _ = mean_sem(b)
    p = pval(a, b)

    # Means ± SEM as points with error bars
    ax.errorbar(gi - dx, m1, yerr=0 if np.isnan(s1) else s1,
                fmt='o', capsize=5, color=c_opto, label="Opto days" if gi == 0 else None)
    ax.errorbar(gi + dx, m2, yerr=0 if np.isnan(s2) else s2,
                fmt='o', capsize=5, color=c_ctrl, label="Control days" if gi == 0 else None)

    # Overlay all days as semi transparent dots in same color
    jitter = 0.06
    for v in a[~np.isnan(a)]:
        ax.scatter((gi - dx) + (rng.random()-0.5)*2*jitter, v, color=c_opto, alpha=alpha_pts, zorder=3)
    for v in b[~np.isnan(b)]:
        ax.scatter((gi + dx) + (rng.random()-0.5)*2*jitter, v, color=c_ctrl, alpha=alpha_pts, zorder=3)

    # p value bracket
    top = np.nanmax([m1 + (0 if np.isnan(s1) else s1), m2 + (0 if np.isnan(s2) else s2)])
    if np.isnan(top): top = 0
    h = 0.06 * (top if top > 0 else 1.0)
    y = top + h
    ax.plot([gi - dx, gi - dx, gi + dx, gi + dx], [y, y + h/3, y + h/3, y], color="black")
    ax.text(gi, y + h/2, f"{stars(p)}  p={p:.4f}" if not np.isnan(p) else "p = NA",
            ha='center', va='bottom', fontsize=10)
    ymax = max(ymax, y + 2*h)

# Bottom axis: no condition labels
ax.set_xticks(group_x)
ax.set_xticklabels([])
ax.set_xlabel("Groups")

# Top axis: condition names
secax = ax.secondary_xaxis('top')
secax.set_xticks(group_x)
secax.set_xticklabels([t for _, t in conds])
secax.set_xlabel("Conditions")

ax.set_ylabel("Average success rate")
ax.set_title("Means ± SEM with all days overlaid")

# Legend at bottom
opt_handle = ax.plot([], [], 'o', color=c_opto, label="Opto days")[0]
ctrl_handle = ax.plot([], [], 'o', color=c_ctrl, label="Control days")[0]
fig.legend(handles=[opt_handle, ctrl_handle],
           loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))

# Day lists at bottom in black
opto_text = "Opto days: " + ", ".join(map(str, opto_days)) if opto_days else "Opto days: —"
ctrl_text = "Control days: " + ", ".join(map(str, control_days)) if control_days else "Control days: —"
fig.text(0.01, 0.01, opto_text, ha='left', va='bottom', fontsize=9, color='black')
fig.text(0.01, 0.04, ctrl_text, ha='left', va='bottom', fontsize=9, color='black')

ax.set_ylim(top=max(ax.get_ylim()[1], ymax))
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()