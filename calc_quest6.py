#!/usr/bin/env python3
"""Calculations for QUEST6 dataset.

Uses UKS, pa-SS-KS, pa-OSS-KS, and pa-STA-KS methods.
"""

import argparse
import json
import os
import time

import yaml
from pyscf import dft, gto
from pyscf.data.nist import HARTREE2EV

from calc_quest1_lowest import (
    eval_and_print_maes,
    fmt_time,
    pyscf_atom_input,
    run_oss,
    run_singlet,
    run_sta,
    run_triplet,
)
from methods.pa_SS_KS import pa_SS_KS
from methods.UKS import UKS
from sets.quest6 import reference, systems


def print_summary(res):
    """Print a summary table of singlet excitation energies and MAEs for all systems.

    Args:
        res: nested dict mapping system name → excitation label → result dict
             with keys "UKS S", "pa-SS-KS S", "pa-OSS-KS S", "pa-STA-KS S",
             "TBE S" (energies in eV, None if unavailable).
    """
    print("\nSUMMARY")

    print("\nSINGLET EXCITATIONS:")
    calc, calc_sa, calc_oss, calc_sta, ref = [], [], [], [], []
    for system, _ in res.items():
        for excitation in res[system]:
            if res[system][excitation]["TBE S"] is not None:
                calc.append(res[system][excitation]["UKS S"])
                calc_sa.append(res[system][excitation]["pa-SS-KS S"])
                calc_oss.append(res[system][excitation]["pa-OSS-KS S"])
                calc_sta.append(res[system][excitation]["pa-STA-KS S"])
                ref.append(res[system][excitation]["TBE S"])
                e = res[system][excitation]["UKS S"]
                e_sa = res[system][excitation]["pa-SS-KS S"]
                e_oss = res[system][excitation]["pa-OSS-KS S"]
                e_sta = res[system][excitation]["pa-STA-KS S"]
                e_ref = res[system][excitation]["TBE S"]
                print(f"{system:30}  {excitation:8}  {e:.2f}  {e_sa:.2f}  {e_oss:.2f}  {e_sta:.2f}  {e_ref:.2f}")

    eval_and_print_maes(calc, calc_sa, calc_oss, calc_sta, ref)


def calc_quest6(config):
    """Run all four DeltaSCF methods on each system in the QUEST6 dataset.

    For each molecule and excitation, performs a ground-state RKS calculation, then runs
    UKS, pa-SS-KS, pa-OSS-KS, and pa-STA-KS excited-state calculations. Only singlet
    excitation energies are stored. Writes per-system results to results.json and prints
    a summary with MAEs against TBE references.

    Args:
        config: dict with keys:
            basis       - basis set name (e.g. "cc-pVDZ")
            auxbasis    - auxiliary basis for density fitting
            xc          - exchange-correlation functional (e.g. "PBE")
            excitations - dict mapping system name to a dict of excitation label → [from, to]
            grid_level  - DFT integration grid level (0-9)
            frac_occ    - if True, use fractional occupations for degenerate orbitals
            systems     - list of system names to run, or None to run all
    """
    print("\nCONFIG")
    for key, value in config.items():
        print(f"  {key}: {value}")

    if config["systems"] is None:
        systems2use = list(systems.keys())
    else:
        systems2use = config["systems"]

    total = sum(len(config["excitations"][s]) for s in systems2use)
    t_total = time.time()
    count = 0
    results = {}
    for system in systems2use:
        results[system] = {}

        for excitation in config["excitations"][system]:
            count += 1
            t_exc = time.time()
            print(f"\n[{count}/{total}] {system} {excitation}", flush=True)

            results[system][excitation] = {
                "excitation": reference[system][excitation]["label"],
                "TBE S": reference[system][excitation]["singlet"]["tbe"],
            }

            mol = gto.M(
                atom=pyscf_atom_input(systems[system]),
                basis=config["basis"],
                charge=systems[system]["charge"],
            )
            mol.verbose = 0
            mol.build()

            # Ground-state singlet
            mf = dft.RKS(mol, xc=config["xc"]).density_fit(auxbasis=config["auxbasis"])
            mf.grids.level = config["grid_level"]
            mf.run()
            if not mf.converged:
                print("RKS ground-state calculation did not converge")
            print("Energy of ground-state singlet:", mf.e_tot, flush=True)
            mf = mf.to_uks()

            exci = config["excitations"][system][excitation]

            mf_mixed_singlet = run_singlet(mf, config["frac_occ"], exci, UKS)
            mf_triplet = run_triplet(mf, config["frac_occ"], exci, UKS)

            mf_mixed_singlet_sa = run_singlet(mf, config["frac_occ"], exci, pa_SS_KS)
            mf_triplet_sa = run_triplet(mf, config["frac_occ"], exci, pa_SS_KS)

            mf_oss = run_oss(mf, config["frac_occ"], exci)

            mf_sta = run_sta(mf, config["frac_occ"], exci)

            exc_s = (2 * mf_mixed_singlet.e_tot - mf_triplet.e_tot - mf.e_tot) * HARTREE2EV
            print(f"Singlet Excitation energy (UKS):       {exc_s:.4f} eV")
            results[system][excitation]["UKS S"] = exc_s

            exc_s = (2 * mf_mixed_singlet_sa.e_tot - mf_triplet_sa.e_tot - mf.e_tot) * HARTREE2EV
            print(f"Singlet Excitation energy (pa-SS-KS):  {exc_s:.4f} eV")
            results[system][excitation]["pa-SS-KS S"] = exc_s

            exc_s = (mf_oss.e_tot - mf.e_tot) * HARTREE2EV
            print(f"Singlet Excitation energy (pa-OSS-KS): {exc_s:.4f} eV")
            results[system][excitation]["pa-OSS-KS S"] = exc_s

            exc_s = (mf_sta.e_tot_oss - mf.e_tot) * HARTREE2EV
            print(f"Singlet Excitation energy (pa-STA-KS): {exc_s:.4f} eV")
            results[system][excitation]["pa-STA-KS S"] = exc_s

            elapsed_exc = fmt_time(time.time() - t_exc)
            elapsed_total = fmt_time(time.time() - t_total)
            print(f"Elapsed: {elapsed_exc}  Total: {elapsed_total}", flush=True)

    with open("results.json", "w", encoding="utf-8") as file_obj:
        print(json.dumps(results, indent=4), file=file_obj)

    print_summary(results)


base_config = {
    "basis": "cc-pVDZ",
    "auxbasis": "aug-cc-pV5Z-RIFIT",
    "xc": "PBE",
    "excitations": {
        "Aminobenzonitrile": {"A1": [0, 1]},
        "Aniline": {"A1": [0, 3]},
        "Azulene": {"A1": [0, 2], "B2": [-1, 2]},
        "Benzonitrile": {"A2": [-2, 1]},
        "Benzothiadiazole": {"B2": [0, 1]},
        "Dimethylaminobenzonitrile": {"A1": [0, 1]},
        "Dimethylaniline": {"B2": [0, 1], "A1": [0, 3]},
        "Hydrogen chloride": {"Pi": [0, 1]},
        "Nitroaniline": {"A1": [0, 1]},
        "Nitrobenzene": {"A1": [-2, 1]},
        "Nitrodimethylaniline": {"A1": [0, 1]},
        "Nitropyridine N-Oxide": {"A1": [0, 1]},
        "N-Phenylpyrrole": {"B2": [0, 1], "A1": [0, 2]},
        "Phthalazine": {"A2": [0, 1], "B1": [0, 2]},
        "Quinoxaline": {"B2": [-1, 1], "A1": [-1, 2], "B1": [0, 3]},
        "Twisted DMABN": {"A2": [0, 1], "B1": [0, 2]},
        "Twisted PP": {"B2": [0, 1], "A1": [0, 2], "A2": [-1, 1], "B1": [-1, 2]},
    },
    "grid_level": 3,
    "frac_occ": True,
    "systems": None,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="quest6", description="quest6 with DFT")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if os.path.exists(args.config):
        print(f"YAML-config at {args.config} was found. Using it.")
        with open(args.config, "r") as file_obj:
            config = yaml.safe_load(file_obj)
    else:
        print(f"{args.config} was not found. base_config is used.")
        config = base_config

    calc_quest6(config)
