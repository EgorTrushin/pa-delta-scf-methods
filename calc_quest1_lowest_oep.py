#!/usr/bin/env python3
"""Calculation of lowest excitations for QUEST1 dataset with OEP methods.

Uses UKS-OEP, SS-KS-OEP, OSS-KS-OEP, and STA-KS-OEP methods.
"""

import argparse
import os
import json
import time
import yaml
import numpy as np
from pyscf import dft, gto
from methods_oep.dftoep import DFTOEP
from methods_oep.osdftoep_swap import OSDFTOEP_swap
from methods_oep.osdftoep_oss_swap import OSDFTOEP_OSS_swap
from methods_oep.osdftoep_sta_swap import OSDFTOEP_STA_swap
from methods_oep.osdftoep_occ import OSDFTOEP_occ
from methods_oep.osdftoep_oss_occ import OSDFTOEP_OSS_occ
from methods_oep.osdftoep_sta_occ import OSDFTOEP_STA_occ
from sets.quest1_lowest import systems, reference
from calc_quest1_lowest import pyscf_atom_input, HA_TO_EV


def eval_and_print_maes(uks, ss, oss, sta, ref_):
    """Compute and print mean absolute errors for all four OEP methods.

    Args:
        uks: list of UKS-OEP excitation energies in eV.
        ss: list of SS-KS-OEP excitation energies in eV.
        oss: list of OSS-KS-OEP excitation energies in eV.
        sta: list of STA-KS-OEP excitation energies in eV.
        ref_: list of TBE reference values in eV.
    """
    uks  = np.array(uks)
    ss   = np.array(ss)
    oss  = np.array(oss)
    sta  = np.array(sta)
    ref_ = np.array(ref_)
    print(f"MAE (UKS-OEP)    = {np.mean(np.abs(uks - ref_)):.2f}")
    print(f"MAE (SS-KS-OEP)  = {np.mean(np.abs(ss  - ref_)):.2f}")
    print(f"MAE (OSS-KS-OEP) = {np.mean(np.abs(oss  - ref_)):.2f}")
    print(f"MAE (STA-KS-OEP) = {np.mean(np.abs(sta  - ref_)):.2f}")


def print_summary(res):
    """Print a summary table of excitation energies and MAEs for all systems.

    Prints three sections — singlet excitations, triplet excitations, and
    singlet-triplet gaps — each followed by MAEs against TBE references.
    Systems without a TBE value for a given multiplicity are skipped.

    Args:
        res: dict mapping system name to a result dict with keys
             "UKS-OEP S/T", "SS-KS-OEP S/T", "OSS-KS-OEP S/T", "STA-KS-OEP S/T",
             "TBE S/T" (excitation energies in eV, None if not available).
    """
    print("\nSUMMARY")

    print("\nSINGLET EXCITATIONS:")
    calc, calc_ss, calc_oss, calc_sta, ref = [], [], [], [], []
    for system, _ in res.items():
        if res[system]["TBE S"] is not None:
            calc.append(res[system]["UKS-OEP S"])
            calc_ss.append(res[system]["SS-KS-OEP S"])
            calc_oss.append(res[system]["OSS-KS-OEP S"])
            calc_sta.append(res[system]["STA-KS-OEP S"])
            ref.append(res[system]["TBE S"])
            e     = res[system]["UKS-OEP S"]
            e_ss  = res[system]["SS-KS-OEP S"]
            e_oss = res[system]["OSS-KS-OEP S"]
            e_sta = res[system]["STA-KS-OEP S"]
            e_ref = res[system]["TBE S"]
            print(f"{system:30}  {e:.2f}  {e_ss:.2f}  {e_oss:.2f}  {e_sta:.2f}  {e_ref:.2f}")

    eval_and_print_maes(calc, calc_ss, calc_oss, calc_sta, ref)

    print("\nTRIPLET EXCITATIONS:")
    calc, calc_ss, calc_oss, calc_sta, ref = [], [], [], [], []
    for system, _ in res.items():
        if res[system]["TBE T"] is not None:
            calc.append(res[system]["UKS-OEP T"])
            calc_ss.append(res[system]["SS-KS-OEP T"])
            calc_oss.append(res[system]["OSS-KS-OEP T"])
            calc_sta.append(res[system]["STA-KS-OEP T"])
            ref.append(res[system]["TBE T"])
            e     = res[system]["UKS-OEP T"]
            e_ss  = res[system]["SS-KS-OEP T"]
            e_oss = res[system]["OSS-KS-OEP T"]
            e_sta = res[system]["STA-KS-OEP T"]
            e_ref = res[system]["TBE T"]
            print(f"{system:30}  {e:.2f}  {e_ss:.2f}  {e_oss:.2f}  {e_sta:.2f}  {e_ref:.2f}")

    eval_and_print_maes(calc, calc_ss, calc_oss, calc_sta, ref)

    print("\nSINGLET-TRIPLET GAPS:")
    calc, calc_ss, calc_oss, calc_sta, ref = [], [], [], [], []
    for system, _ in res.items():
        if res[system]["TBE S"] is not None and res[system]["TBE T"] is not None:
            st     = res[system]["UKS-OEP S"]     - res[system]["UKS-OEP T"]
            st_ss  = res[system]["SS-KS-OEP S"]   - res[system]["SS-KS-OEP T"]
            st_oss = res[system]["OSS-KS-OEP S"]  - res[system]["OSS-KS-OEP T"]
            st_sta = res[system]["STA-KS-OEP S"]  - res[system]["STA-KS-OEP T"]
            st_ref = res[system]["TBE S"]          - res[system]["TBE T"]
            print(f"{system:30}  {st:.2f}  {st_ss:.2f}  {st_oss:.2f}  {st_sta:.2f}  {st_ref:.2f}")
            calc.append(st)
            calc_ss.append(st_ss)
            calc_oss.append(st_oss)
            calc_sta.append(st_sta)
            ref.append(st_ref)

    eval_and_print_maes(calc, calc_ss, calc_oss, calc_sta, ref)


def calc_quest1(config):
    """Run all four OEP methods on each system in the QUEST1 dataset.

    For each molecule, performs a ground-state RKS then KS-OEP calculation,
    then runs UKS-OEP, SS-KS-OEP, OSS-KS-OEP, and STA-KS-OEP excited-state
    calculations. Writes per-system results to results.json and prints a summary
    with MAEs against TBE reference values.

    Args:
        config: dict with keys:
            basis        - orbital basis set name (e.g. "aug-cc-pVQZ")
            oep_basis    - auxiliary basis for OEP equation
            dfit_basis   - auxiliary basis for density fitting
            xc           - exchange-correlation functional (e.g. "PBE")
            excitation   - default [from, to] excitation indices (1-based)
            exceptions   - dict mapping system name to a custom excitation index
            grid_level   - DFT integration grid level (0-9)
            space_sym    - if True, enforce spatial symmetry in OEP
            thr_fai_oep  - OEP threshold
            conv_thr     - energy convergence threshold
            swap         - if True, use orbital-swap; else use occupation numbers
            systems      - list of system names to run, or None to run all
    """
    print("\nCONFIG")
    for key, value in config.items():
        print(f"  {key}: {value}")

    if config["systems"] is None:
        systems2use = list(systems.keys())
    else:
        systems2use = config["systems"]

    t_total = time.time()
    results = {}
    for i, system in enumerate(systems2use, 1):
        t_mol = time.time()
        print(f"\n[{i}/{len(systems2use)}] {system}", flush=True)

        results[system] = {
            "excitation": reference[system]["label"],
            "TBE S": reference[system]["singlet"]["tbe"],
            "TBE T": reference[system]["triplet"]["tbe"],
        }

        mol = gto.M(
            atom=pyscf_atom_input(systems[system]),
            basis=config["basis"],
            charge=systems[system]["charge"],
        )
        mol.verbose = 0
        mol.symmetry = False
        mol.build()

        print("Running ground-state KS calculation")
        mf = dft.RKS(mol, xc=config["xc"]).density_fit(auxbasis=config["dfit_basis"])
        mf.grids.level = config["grid_level"]
        mf.run()
        print("Energy of standard DFT calculation:", mf.e_tot)

        print("\nRunning ground-state KS-OEP calculation")
        mf_oep = DFTOEP(mf, config["oep_basis"], space_sym=config["space_sym"])
        mf_oep.run(maxit=50, thr_fai_oep=config["thr_fai_oep"], e_conv_thr=config["conv_thr"])
        E_GS = mf_oep.e_tot
        print("Energy of ground-state KS-OEP calculation:", E_GS)

        mf = mf.to_uks()

        exci = config["exceptions"][system] if system in config["exceptions"] else config["excitation"]

        occ = mf.mo_occ.copy()
        occ[0][mf.nelec[0] + exci[0] - 1] = 0
        occ[0][mf.nelec[0] + exci[1] - 1] = 1

        occ3 = mf.mo_occ.copy()
        occ3[1][mf.nelec[1] + exci[0] - 1] = 0
        occ3[0][mf.nelec[0] + exci[1] - 1] = 1

        cls_single = OSDFTOEP_swap if config["swap"] else OSDFTOEP_occ
        cls_pair   = OSDFTOEP_OSS_swap if config["swap"] else OSDFTOEP_OSS_occ
        cls_sta    = OSDFTOEP_STA_swap if config["swap"] else OSDFTOEP_STA_occ

        print("\nRunning excited-state singlet UKS-OEP calculation")
        mf_oep2 = cls_single(mf, config["oep_basis"], occ, spin_sym=False, space_sym=config["space_sym"])
        mf_oep2.run(maxit=50, thr_fai_oep=config["thr_fai_oep"], e_conv_thr=config["conv_thr"])
        mf_oep2.print_occ_numbers()
        E_SINGLET = mf_oep2.e_tot
        print("Energy of excited-state singlet UKS-OEP:", E_SINGLET)

        print("\nRunning excited-state singlet SS-KS-OEP calculation")
        mf_oep2 = cls_single(mf, config["oep_basis"], occ, spin_sym=True, space_sym=config["space_sym"])
        mf_oep2.run(maxit=50, thr_fai_oep=config["thr_fai_oep"], e_conv_thr=config["conv_thr"])
        mf_oep2.print_occ_numbers()
        E_SINGLET_SS = mf_oep2.e_tot
        print("Energy of excited-state singlet SS-KS-OEP:", E_SINGLET_SS)

        print("\nRunning excited-state triplet UKS-OEP calculation")
        mf_oep2 = cls_single(mf, config["oep_basis"], occ3, spin_sym=False, space_sym=config["space_sym"])
        mf_oep2.run(maxit=50, thr_fai_oep=config["thr_fai_oep"], e_conv_thr=config["conv_thr"])
        mf_oep2.print_occ_numbers()
        E_TRIPLET = mf_oep2.e_tot
        print("Energy of excited-state triplet UKS-OEP:", E_TRIPLET)

        print("\nRunning excited-state triplet SS-KS-OEP calculation")
        mf_oep2 = cls_single(mf, config["oep_basis"], occ3, spin_sym=True, space_sym=config["space_sym"])
        mf_oep2.run(maxit=50, thr_fai_oep=config["thr_fai_oep"], e_conv_thr=config["conv_thr"])
        mf_oep2.print_occ_numbers()
        E_TRIPLET_SS = mf_oep2.e_tot
        print("Energy of excited-state triplet SS-KS-OEP:", E_TRIPLET_SS)

        print("\nRunning OSS-KS-OEP calculation")
        mf_oep2 = cls_pair(mf, config["oep_basis"], occ, occ3, space_sym=config["space_sym"])
        mf_oep2.run(maxit=50, thr_fai_oep=config["thr_fai_oep"], e_conv_thr=config["conv_thr"])
        mf_oep2.print_occ_numbers()
        E_SINGLET_OSS = mf_oep2.e_tot
        E_TRIPLET_OSS = mf_oep2.e_tot3
        print("Energy of excited-state singlet OSS-KS-OEP:", E_SINGLET_OSS)
        print("Energy of excited-state triplet OSS-KS-OEP:", E_TRIPLET_OSS)

        print("\nRunning STA-KS-OEP calculation")
        mf_oep2 = cls_sta(mf, config["oep_basis"], occ, occ3, space_sym=config["space_sym"])
        mf_oep2.run(maxit=50, thr_fai_oep=config["thr_fai_oep"], e_conv_thr=config["conv_thr"])
        mf_oep2.print_occ_numbers()
        E_SINGLET_STA = mf_oep2.e_tot
        E_TRIPLET_STA = mf_oep2.e_tot3
        print("Energy of excited-state singlet STA-KS-OEP:", E_SINGLET_STA)
        print("Energy of excited-state triplet STA-KS-OEP:", E_TRIPLET_STA)

        exc_s = (2 * E_SINGLET - E_TRIPLET - E_GS) * HA_TO_EV
        exc_t = (E_TRIPLET - E_GS) * HA_TO_EV
        print(f"Singlet Excitation energy (UKS-OEP):    {exc_s:.4f} eV")
        print(f"Triplet Excitation energy (UKS-OEP):    {exc_t:.4f} eV")
        results[system]["UKS-OEP S"] = exc_s
        results[system]["UKS-OEP T"] = exc_t

        exc_s = (2 * E_SINGLET_SS - E_TRIPLET_SS - E_GS) * HA_TO_EV
        exc_t = (E_TRIPLET_SS - E_GS) * HA_TO_EV
        print(f"Singlet Excitation energy (SS-KS-OEP):  {exc_s:.4f} eV")
        print(f"Triplet Excitation energy (SS-KS-OEP):  {exc_t:.4f} eV")
        results[system]["SS-KS-OEP S"] = exc_s
        results[system]["SS-KS-OEP T"] = exc_t

        exc_s = (E_SINGLET_OSS - E_GS) * HA_TO_EV
        print(f"Singlet Excitation energy (OSS-KS-OEP): {exc_s:.4f} eV")
        results[system]["OSS-KS-OEP S"] = exc_s
        results[system]["OSS-KS-OEP T"] = exc_t  # from previous SS-KS-OEP triplet calculation

        exc_s = (E_SINGLET_STA - E_GS) * HA_TO_EV
        exc_t = (E_TRIPLET_STA - E_GS) * HA_TO_EV
        print(f"Singlet Excitation energy (STA-KS-OEP): {exc_s:.4f} eV")
        print(f"Triplet Excitation energy (STA-KS-OEP): {exc_t:.4f} eV")
        results[system]["STA-KS-OEP S"] = exc_s
        results[system]["STA-KS-OEP T"] = exc_t

        elapsed_mol   = time.strftime('%H:%M:%S', time.gmtime(time.time() - t_mol))
        elapsed_total = time.strftime('%H:%M:%S', time.gmtime(time.time() - t_total))
        print(f"Elapsed: {elapsed_mol}  Total: {elapsed_total}", flush=True)

    with open("results.json", "w", encoding="utf-8") as file_obj:
        print(json.dumps(results, indent=4), file=file_obj)

    print_summary(results)


base_config = {
    "basis": "aug-cc-pVQZ",
    "oep_basis": "aug-cc-pVDZ-RIFIT",
    "dfit_basis": "aug-cc-pV5Z-RIFIT",
    "xc": "PBE",
    "excitation": [0, 1],
    "exceptions": {"Hydrogen sulfide": [0, 2]},
    "grid_level": 3,
    "systems": None,
    "space_sym": False,
    "thr_fai_oep": 1.7e-2,
    "conv_thr": 1e-8,
    "swap": True,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="quest1_oep", description="quest1 with OEP methods")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if os.path.exists(args.config):
        print(f"YAML-config at {args.config} was found. Using it.")
        with open(args.config, "r") as file_obj:
            config = yaml.safe_load(file_obj)
    else:
        print(f"{args.config} was not found. base_config is used.")
        config = base_config

    calc_quest1(config)
