"""
Design PGA (PGA_d) and value of information from a soil investigation.

Chain of calculations:
    Vs30 -> site hazard -> expected annual loss (EAL) -> expected future loss (EFL)
    -> + seismic design cost -> combined cost -> PGA_d with minimum cost -> value of information

All PGA values are in g.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Inputs taken from the original analysis
# ---------------------------------------------------------------------------
# ESHM20 reference-rock hazard curve: probability of exceedance (PoE) in 50 years
PGA_REF = np.array([
    0.0005, 0.0007184440843, 0.001032323804, 0.001483333861,
    0.002131384874, 0.003062561708, 0.004400558684, 0.006323110708,
    0.009085602964, 0.0130549954, 0.01875856843, 0.02695396504,
    0.03872983346, 0.05565043947, 0.07996345805, 0.1148985468,
    0.1650963624, 0.2372250099, 0.3408658099, 0.4897860494,
    0.7037677794, 1.011235596, 1.453032463, 2.087845154,
    3.0,
])

POE_REF = np.array(
    [1,1,1,1,1,1,1,1,0.9999999404,
    0.9999958873,0.9998915792,
    0.9985356331,0.9888771772,
    0.9491682649,0.8475667834,
    0.6741468906,0.4643102884,
    0.2745105624,0.1399666518,
    0.06220382079,0.02428352833,
    0.008349335752,0.002473960863,
    0.0006161157507,0.0001224722218
])


V_REF = 800          # reference Vs30 [m/s]
GAMMA = -0.3019      # site amplification coefficient

# The four vulnerability curves (columns of the vulnerability CSV), one per design level PGA_d
VULN_COLUMNS = ["EAL_pga0", "EAL_pga01", "EAL_pga03", "EAL_pga05"]
PGAD_VALUES = np.array([0.0, 0.1, 0.3, 0.5])
BUILDING_VALUES = np.array([655840, 656778, 675161,693205])#[2055840, 2056778, 2075161, 2093205])   # converts loss ratio to EUR

PGAD_GRID = np.linspace(0, 1.5, 500)                                # PGA_d values for the cost curves
PGAD_DECISIONS = np.array([0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5])   # PGA_d options in the decision


# ---------------------------------------------------------------------------
# 1. Site hazard
# ---------------------------------------------------------------------------
_E = 1 - 1e-6
_spline = UnivariateSpline(np.log(PGA_REF), np.log(np.clip(POE_REF, 0, _E)), k=3, s=0)


def reference_maf(pga):
    """Mean annual frequency (MAF) of exceeding pga on reference rock."""
    poe = np.clip(np.exp(_spline(np.log(pga))), 1e-10, _E)
    return -np.log(1 - poe) / 50


def amplification_factor(vs30):
    """Site amplification Fs for a given Vs30 [m/s]."""
    return (np.asarray(vs30, dtype=float) / V_REF) ** GAMMA


def site_maf(pga, vs30):
    """MAF of exceeding pga at a site with the given Vs30."""
    return reference_maf(np.asarray(pga, dtype=float) / amplification_factor(vs30))


# ---------------------------------------------------------------------------
# 2. Vulnerability and expected annual loss (EAL)
# ---------------------------------------------------------------------------
def load_vulnerability(path):
    """Read the vulnerability CSV (column 'PGA' plus the four EAL_pga... columns)."""
    return pd.read_csv(path)


def vulnerability_function(pga, vulnerability, column):
    """Loss ratio at pga for one vulnerability curve (linear interpolation)."""
    values = vulnerability[column].values
    f = interp1d(vulnerability["PGA"].values, values, kind="linear",
                 bounds_error=False, fill_value=(0, values[-1]))
    return f(pga)


def calculate_eals(vs30, vulnerability, pga_min=0.2, pga_max=1.9, n=2000):
    """EAL (loss ratio per year) for each vulnerability curve at a site. Returns a dict."""
    pga = np.linspace(pga_min, pga_max, n)
    d_maf = -np.diff(site_maf(pga, vs30))
    eals = {}
    for column in VULN_COLUMNS:
        v = vulnerability_function(pga, vulnerability, column)
        eals[column] = np.sum((v[:-1] + v[1:]) / 2 * d_maf)
    return eals


# ---------------------------------------------------------------------------
# 3. EAL and EFL as functions of PGA_d
# ---------------------------------------------------------------------------
def eal_function(pga_d, B, C):
    """EAL in EUR as a function of PGA_d: B * exp(-C * PGA_d)."""
    return B * np.exp(-C * np.asarray(pga_d, dtype=float))


def fit_eal_function(eals):
    """Fit B and C to the four EAL values. Returns (B, C)."""
    values = np.array([eals[c] for c in VULN_COLUMNS]) * BUILDING_VALUES
    (B, C), _ = curve_fit(eal_function, PGAD_VALUES, values, p0=[values[0], 1.0],
                          bounds=([0, 0], [np.inf, np.inf]))
    return B, C


def efl_function(pga_d, B, C, t=50, discount=0.03):
    """Expected future loss over t years with the given discount rate."""
    return eal_function(pga_d, B, C) * (1 - np.exp(-discount * t)) / discount


def seismic_design_cost(pga_d):
    """Extra construction cost in EUR for designing to pga_d."""
    pga_d = np.asarray(pga_d, dtype=float)
    return np.where(pga_d < 0.1, 12820 * pga_d, 90542.4 * pga_d - 7757.5)


# ---------------------------------------------------------------------------
# 4. Combined cost for one site
# ---------------------------------------------------------------------------
def cost_curves(vs30, vulnerability, pga_d=PGAD_GRID, t=50, discount=0.03):
    """EFL, design cost and combined cost at each pga_d. Returns a DataFrame."""
    B, C = fit_eal_function(calculate_eals(vs30, vulnerability))
    efl = efl_function(pga_d, B, C, t, discount)
    design = seismic_design_cost(pga_d)
    return pd.DataFrame({"PGAd": pga_d, "EFL": efl, "Design_cost": design, "Combined": efl + design})


def minimum_cost(curves):
    """PGA_d with the lowest combined cost, and that cost. Returns (pga_d, cost)."""
    row = curves.loc[curves["Combined"].idxmin()]
    return float(row["PGAd"]), float(row["Combined"])


# ---------------------------------------------------------------------------
# 5. Soil profiles
# ---------------------------------------------------------------------------
def generate_profiles(n=10, vs1_range=(100, 300), vs2_range=(200, 800), thickness=(10, 20), seed=42):
    """Random two-layer Vs profiles within the given ranges. Returns a DataFrame with Vs30."""
    d1, d2 = thickness
    if d1 + d2 != 30:
        raise ValueError("Layer thicknesses must add up to 30 m")
    np.random.seed(seed)
    theta1 = np.random.uniform(0, 1, n)
    theta2 = np.random.uniform(0, 1, n)
    vs1 = vs1_range[0] + theta1 * (vs1_range[1] - vs1_range[0])
    vs2 = vs2_range[0] + theta2 * (vs2_range[1] - vs2_range[0])
    vs30 = 30 / (d1 / vs1 + d2 / vs2)
    return pd.DataFrame({"Profile": np.arange(1, n + 1), "Vs1": vs1, "Vs2": vs2,
                         "Thickness1": d1, "Thickness2": d2, "Vs30": vs30})


def profile_cost_curves(profiles, vulnerability, pga_d=PGAD_GRID, t=50, discount=0.03):
    """Combined cost curve for every profile. Returns an array (profiles x pga_d)."""
    return np.array([cost_curves(v, vulnerability, pga_d, t, discount)["Combined"].to_numpy()
                     for v in profiles["Vs30"]])


def profile_results(profiles, combined_curves, pga_d=PGAD_GRID):
    """Optimal PGA_d and minimum cost for every profile. Returns a DataFrame."""
    i = combined_curves.argmin(axis=1)
    results = profiles[["Profile", "Vs1", "Vs2", "Vs30"]].copy()
    results["PGAd_min"] = pga_d[i]
    results["Minimum_cost"] = combined_curves[np.arange(len(i)), i]
    return results


# ---------------------------------------------------------------------------
# 6. Value of information
# ---------------------------------------------------------------------------
def cost_matrix(profiles, vulnerability, decisions=PGAD_DECISIONS, t=50, discount=0.03):
    """Total cost of every decision (columns) for every possible true profile (rows)."""
    return np.array([cost_curves(v, vulnerability, decisions, t, discount)["Combined"].to_numpy()
                     for v in profiles["Vs30"]])


def value_of_information(costs, decisions=PGAD_DECISIONS):
    """
    Value of perfect information from a cross-hole investigation.

    costs : cost matrix (profiles x decisions); all profiles are assumed equally likely.
    Without information one decision is chosen for all profiles; with perfect information
    the best decision is chosen for the true profile. VoI is the difference in expected cost.
    """
    expected = costs.mean(axis=0)
    i = expected.argmin()
    min_each = costs.min(axis=1)
    cost_with_info = min_each.mean()
    return {"expected_cost_per_decision": expected,
            "optimal_without_info": decisions[i],
            "cost_without_info": expected[i],
            "optimal_per_profile": decisions[costs.argmin(axis=1)],
            "min_cost_per_profile": min_each,
            "cost_with_info": cost_with_info,
            "VoI": expected[i] - cost_with_info}


# ---------------------------------------------------------------------------
# Plots (call plt.show() afterwards)
# ---------------------------------------------------------------------------
def plot_vulnerability(vulnerability, pga_min=0.2, pga_max=1.9):
    pga = np.linspace(pga_min, pga_max, 500)
    for column in VULN_COLUMNS:
        plt.plot(pga, vulnerability_function(pga, vulnerability, column), label=column)
    plt.xlabel("PGA [g]"); plt.ylabel("Loss ratio")
    plt.xlim(pga_min, pga_max); plt.ylim(0, 1)
    plt.grid(alpha=0.3); plt.legend()


def plot_site_hazard(vs30_values):
    pga = np.logspace(-3, np.log10(3), 500)
    for vs30 in np.atleast_1d(vs30_values):
        plt.loglog(pga, site_maf(pga, vs30), label=f"Vs30 = {vs30:.0f} m/s")
    plt.xlabel("PGA [g]"); plt.ylabel("MAF of exceedance")
    plt.grid(which="both", alpha=0.3); plt.legend()


def plot_cost_curves(curves, vs30=None):
    plt.plot(curves["PGAd"], curves["Design_cost"], label="Seismic design cost")
    plt.plot(curves["PGAd"], curves["EFL"], label="EFL")
    plt.plot(curves["PGAd"], curves["Combined"], label="Combined")
    pga_d_min, cost_min = minimum_cost(curves)
    plt.scatter(pga_d_min, cost_min, s=100, color="tab:green", zorder=5, label="Minimum combined cost")
    plt.annotate(f"{pga_d_min:.2f} g", xy=(pga_d_min, cost_min), xytext=(0, 12),
                 textcoords="offset points", ha="center")
    plt.xlabel(r"$PGA_d$ [g]"); plt.ylabel("Cost [€]")
    plt.xlim(0, 0.8); plt.ylim(bottom=0)
    if vs30 is not None:
        plt.title(f"Vs30 = {vs30:.0f} m/s")
    plt.grid(alpha=0.3); plt.legend()


def plot_profiles(profiles, depth=30):
    for p in profiles.itertuples():
        z = [0, p.Thickness1, p.Thickness1, depth]
        vs = [p.Vs1, p.Vs1, p.Vs2, p.Vs2]
        plt.step(vs, z, where="post", linewidth=2, alpha=0.7, label=f"Profile {p.Profile}")
    plt.axhline(depth, linestyle="--", linewidth=2, label="Bedrock")
    plt.xlabel(r"$V_S$ [m/s]"); plt.ylabel("Depth [m]")
    plt.ylim(depth, 0); plt.grid(alpha=0.3); plt.legend(fontsize=9, ncol=2)


def plot_profile_cost_curves(profiles, combined_curves, pga_d=PGAD_GRID):
    for vs30, curve in zip(profiles["Vs30"], combined_curves):
        line, = plt.plot(pga_d, curve, linewidth=2, alpha=0.7, label=f"Vs30 = {vs30:.0f} m/s")
        i = curve.argmin()
        plt.scatter(pga_d[i], curve[i], marker="x", s=60, linewidths=2, color=line.get_color(), zorder=5)
    plt.xlabel(r"$PGA_d$ [g]"); plt.ylabel("Combined cost [€]")
    plt.xlim(0, 0.8); plt.grid(alpha=0.3); plt.legend(fontsize=9, ncol=2)


def plot_average_cost(combined_curves, costs, decisions=PGAD_DECISIONS, pga_d=PGAD_GRID):
    plt.plot(pga_d, combined_curves.mean(axis=0), linewidth=3, label="Average over profiles")
    expected = costs.mean(axis=0)
    plt.scatter(decisions, expected, s=40, zorder=5, label="Decision options")
    i = expected.argmin()
    plt.scatter(decisions[i], expected[i], marker="x", s=100, linewidths=3, color="tab:red",
                zorder=6, label=f"Optimal: {decisions[i]:.2f} g")
    plt.xlabel(r"$PGA_d$ [g]"); plt.ylabel("Average combined cost [€]")
    plt.xlim(0, 0.8); plt.grid(alpha=0.3); plt.legend()
