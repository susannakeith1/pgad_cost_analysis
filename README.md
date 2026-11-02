# Design PGA (PGA_d) cost analysis

Finds the design PGA (PGA_d) with the lowest total cost, and the value of a soil investigation (cross-hole survey).

Total cost = expected future loss (EFL) + seismic design cost.

## Use

1. Put the vulnerability CSV `Gkimprixis_PGAd_vulnerbility.csv` in this folder. It needs the columns `PGA`, `EAL_pga0`, `EAL_pga01`, `EAL_pga03`, `EAL_pga05`.
2. Install the packages: `pip install numpy pandas scipy matplotlib jupyter`
3. Open `pgad_cost_analysis.ipynb` and run the cells in order. It explains each step.

## Files

| File | Contents |
|---|---|
| `pgad_cost.py` | The functions |
| `pgad_cost_analysis.ipynb` | Worked example, one section per step |
| `Gkimprixis_PGAd_vulnerbility.csv` | Vulnerability data (add this yourself) |

## Functions

| Function | What it does |
|---|---|
| `site_maf(pga, vs30)` | Annual frequency of exceeding a PGA at a site |
| `calculate_eals(vs30, vulnerability)` | Expected annual loss for the four design levels |
| `fit_eal_function(eals)` | Fits EAL as a function of PGA_d: B exp(-C PGA_d) |
| `efl_function(pga_d, B, C)` | Expected future loss (50 years, 5 % discount rate) |
| `seismic_design_cost(pga_d)` | Extra construction cost of designing to PGA_d |
| `cost_curves(vs30, vulnerability)` | EFL, design cost and combined cost at each PGA_d |
| `minimum_cost(curves)` | PGA_d with the lowest combined cost |
| `generate_profiles(...)` | Random two-layer Vs profiles, with Vs30 |
| `profile_cost_curves(...)` / `profile_results(...)` | Cost curves and optimum for every profile |
| `cost_matrix(...)` / `value_of_information(...)` | Value of perfect information |

## Notes

* PGA values are in g, Vs30 in m/s, costs in EUR.
* The hazard curve, building values, design cost function, discount rate (5 %) and time horizon (50 years) are the values from the original analysis. They are set at the top of `pgad_cost.py`, or as arguments (`t`, `discount`).
* The EAL only includes PGA between 0.2 and 1.9 g, the range of the vulnerability data.
* All profiles are assumed equally likely. The value of information is for perfect information.
