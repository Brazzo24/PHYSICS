"""
main.py
=======
Entry point for the Bicycle Dynamics & Control simulation project.

Runs all analysis scenarios across the three model levels and saves
figures to ./figures/.

Usage
-----
    python main.py                  # run everything
    python main.py --level 3        # only Level-3 (Whipple) analyses
    python main.py --scenario B3    # one specific scenario
    python main.py --show           # open figures interactively after saving

Scenario index
--------------
  Stability analysis  (analysis/stability.py)
    A1  Level-1 open-loop poles & min P-gain
    A2  Level-2 front-fork gains k₁(V), k₂(V)
    A3  Level-3 root locus vs velocity          [paper Fig. 8]
    A4  Level-3 real parts of poles vs velocity [paper Fig. 9]

  Step / time-domain  (analysis/step_response.py)
    B1  Level-1 open-loop roll instability
    B2  Level-1 P-control comparison
    B3  Level-2 inverse response (handlebar torque step) [paper Fig. 5]
    B4  Level-3 inverse response (Whipple model)
    B5  Level-3 free roll above/below critical speed

  Self-stabilisation  (analysis/self_stabilization.py)
    C1  Critical velocity vs trail and head angle
    C2  Stable window with vs without rider
    C3  Gyroscopic effect (front wheel inertia) [paper Fig. 13]
    C4  Summary printout of critical velocities
"""
import sys
import argparse
import matplotlib
matplotlib.use("Agg")          # non-interactive backend by default
import matplotlib.pyplot as plt

# ── make project root importable ──────────────────────────────────────────────
import os
sys.path.insert(0, os.path.dirname(__file__))

from analysis.stability          import (plot_level1_stability,
                                         plot_level2_front_fork_gains,
                                         plot_level3_root_locus,
                                         plot_level3_real_parts)
from analysis.step_response      import (plot_B1_openloop,
                                         plot_B2_pcontrol,
                                         plot_B3_inverse_response_L2,
                                         plot_B4_inverse_response_L3,
                                         plot_B5_free_roll)
from analysis.self_stabilization import (plot_C1_Vc_geometry,
                                         plot_C2_stable_window_rider,
                                         plot_C3_gyroscopic_effect,
                                         print_C4_summary)


SCENARIOS = {
    # ── Stability ─────────────────────────────────────────────────────────────
    "A1": ("Level-1 stability (poles & min gain)",          plot_level1_stability),
    "A2": ("Level-2 front-fork gains",                      plot_level2_front_fork_gains),
    "A3": ("Level-3 root locus  [paper Fig. 8]",            plot_level3_root_locus),
    "A4": ("Level-3 real parts vs velocity [paper Fig. 9]", plot_level3_real_parts),
    # ── Step response ──────────────────────────────────────────────────────────
    "B1": ("Level-1 open-loop instability",                 plot_B1_openloop),
    "B2": ("Level-1 P-control",                             plot_B2_pcontrol),
    "B3": ("Level-2 inverse response [paper Fig. 5]",       plot_B3_inverse_response_L2),
    "B4": ("Level-3 inverse response (Whipple)",            plot_B4_inverse_response_L3),
    "B5": ("Level-3 free roll: above/below Vc",             plot_B5_free_roll),
    # ── Self-stabilisation ─────────────────────────────────────────────────────
    "C1": ("Critical velocity vs geometry",                 plot_C1_Vc_geometry),
    "C2": ("Stable window with/without rider",              plot_C2_stable_window_rider),
    "C3": ("Gyroscopic effect [paper Fig. 13]",             plot_C3_gyroscopic_effect),
    "C4": ("Critical velocity summary (console)",           print_C4_summary),
}

LEVEL_MAP = {
    1: ["A1", "B1", "B2"],
    2: ["A2", "B3", "C1"],
    3: ["A3", "A4", "B4", "B5", "C2", "C3", "C4"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Bicycle Dynamics & Control — simulation runner")
    parser.add_argument("--scenario", "-s", default=None,
                        help="Run a single scenario (e.g. B3)")
    parser.add_argument("--level", "-l", type=int, choices=[1, 2, 3],
                        default=None,
                        help="Run all scenarios for a given model level")
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively after saving")
    parser.add_argument("--list", action="store_true",
                        help="List all available scenarios and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable scenarios:")
        for key, (desc, _) in SCENARIOS.items():
            print(f"  {key}  —  {desc}")
        return

    # Determine which scenarios to run
    if args.scenario:
        keys = [args.scenario.upper()]
    elif args.level:
        keys = LEVEL_MAP[args.level]
    else:
        keys = list(SCENARIOS.keys())

    if args.show:
        matplotlib.use("TkAgg")   # switch to interactive if requested

    print("=" * 60)
    print(" Bicycle Dynamics & Control  —  Åström, Klein, Lennartsson 2005")
    print("=" * 60)

    for key in keys:
        if key not in SCENARIOS:
            print(f"  [!] Unknown scenario: {key}")
            continue
        desc, fn = SCENARIOS[key]
        print(f"\n[{key}] {desc}")
        fn()

    # Save path summary
    from utils.plotting import FIGURE_DIR
    n_figs = len([f for f in os.listdir(FIGURE_DIR) if f.endswith(".png")])
    print(f"\n✓  Done.  {n_figs} figure(s) saved to  '{FIGURE_DIR}/'")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
