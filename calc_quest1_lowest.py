#!/usr/bin/env python3
"""Calculation of lowest excitations for QUEST1 dataset.

Uses UKS, pa-SS-KS, pa-OSS-KS, and pa-STA-KS methods.
"""

import argparse
import json
import os
import time

import numpy as np
import yaml
from pyscf import dft, gto

from methods.pa_OSS_KS import pa_OSS_KS
from methods.pa_SS_KS import pa_SS_KS
from methods.pa_STA_KS import pa_STA_KS
from methods.UKS import UKS
from sets.quest1_lowest import reference, systems

HA_TO_EV = 27.2114


def fmt_time(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def pyscf_atom_input(mol_dict):
    """Return atom list for PySCF mol.atom from a molecule dictionary.

    Args:
        mol_dict: dict with keys "atoms" (list of element symbols) and
                  "coords" (list of [x, y, z] coordinates in Angstrom).
    """
    atoms, xyz = mol_dict["atoms"], mol_dict["coords"]
    atom_input = []
    for i, _ in enumerate(atoms):
        atom_input.append([atoms[i], tuple(xyz[i])])
    return atom_input


def run_singlet(mf_gs, frac_occ, excitation, cls):
    """Run an open-shell singlet excited-state calculation.

    Promotes an alpha electron from HOMO+excitation[0]-1 to LUMO+excitation[1]-1
    and runs SCF with the given KS class.

    Args:
        mf_gs: converged UKS ground-state object.
        frac_occ: if True, use fractional occupation numbers for degenerate orbitals.
        excitation: [from, to] indices relative to HOMO/LUMO (1-based).
        cls: KS class to use (e.g. UKS or pa_SS_KS).
    """
    occ = mf_gs.mo_occ.copy()
    occ[0][mf_gs.nelec[0] + excitation[0] - 1] = 0
    occ[0][mf_gs.nelec[0] + excitation[1] - 1] = 1
    label = cls.__name__.replace("_", "-")
    mf_es = cls(mf_gs, occ, frac_occ)
    mf_es.run(verb=False)
    if not mf_es.converged:
        print(f"{label} excited-state singlet calculation did not converge")
    print(f"Energy of excited-state singlet ({label}):", mf_es.e_tot, flush=True)
    mf_es.print_occ_numbers()
    return mf_es


def run_triplet(mf_gs, frac_occ, excitation, cls):
    """Run a triplet excited-state calculation.

    Removes a beta electron from HOMO+excitation[0]-1 and adds an alpha
    electron at LUMO+excitation[1]-1, then runs SCF with the given KS class.

    Args:
        mf_gs: converged UKS ground-state object.
        frac_occ: if True, use fractional occupation numbers for degenerate orbitals.
        excitation: [from, to] indices relative to HOMO/LUMO (1-based).
        cls: KS class to use (e.g. UKS or pa_SS_KS).
    """
    occ = mf_gs.mo_occ.copy()
    occ[1][mf_gs.nelec[1] + excitation[0] - 1] = 0
    occ[0][mf_gs.nelec[0] + excitation[1] - 1] = 1
    label = cls.__name__.replace("_", "-")
    mf_es = cls(mf_gs, occ, frac_occ)
    mf_es.run(verb=False)
    if not mf_es.converged:
        print(f"{label} excited-state triplet calculation did not converge")
    print(f"Energy of excited-state triplet ({label}):", mf_es.e_tot, flush=True)
    mf_es.print_occ_numbers()
    return mf_es


def run_oss(mf_gs, frac_occ, excitation):
    """Run a pa-OSS-KS singlet excited-state calculation.

    Constructs open-shell singlet (occ1) and triplet (occ3) occupation numbers
    for the given excitation and optimizes the OSS composite energy
    E_S = 2*E_M - E_T in a single SCF. Returns the converged object whose
    e_tot holds the singlet energy directly.

    Args:
        mf_gs: converged UKS ground-state object.
        frac_occ: if True, use fractional occupation numbers for degenerate orbitals.
        excitation: [from, to] indices relative to HOMO/LUMO (1-based).
    """
    occ1 = mf_gs.mo_occ.copy()
    occ1[0][mf_gs.nelec[0] + excitation[0] - 1] = 0
    occ1[0][mf_gs.nelec[0] + excitation[1] - 1] = 1
    occ3 = mf_gs.mo_occ.copy()
    occ3[1][mf_gs.nelec[1] + excitation[0] - 1] = 0
    occ3[0][mf_gs.nelec[0] + excitation[1] - 1] = 1
    mf_es = pa_OSS_KS(mf_gs, occ1, occ3, frac_occ)
    mf_es.run(verb=False)
    if not mf_es.converged:
        print("pa-OSS-KS excited-state calculation did not converge")
    print("Energy of excited-state singlet (pa-OSS-KS):", mf_es.e_tot)
    mf_es.print_occ_numbers()
    return mf_es


def run_sta(mf_gs, frac_occ, excitation):
    """Run a pa-STA-KS excited-state calculation.

    Constructs open-shell singlet (occ1) and triplet (occ3) occupation numbers
    and optimizes a state-averaged potential for both states in a single SCF.
    The converged object exposes e_tot_oss (singlet via OSS formula) and e_tot_t
    (triplet energy).

    Args:
        mf_gs: converged UKS ground-state object.
        frac_occ: if True, use fractional occupation numbers for degenerate orbitals.
        excitation: [from, to] indices relative to HOMO/LUMO (1-based).
    """
    occ1 = mf_gs.mo_occ.copy()
    occ1[0][mf_gs.nelec[0] + excitation[0] - 1] = 0
    occ1[0][mf_gs.nelec[0] + excitation[1] - 1] = 1
    occ3 = mf_gs.mo_occ.copy()
    occ3[1][mf_gs.nelec[1] + excitation[0] - 1] = 0
    occ3[0][mf_gs.nelec[0] + excitation[1] - 1] = 1
    mf_es = pa_STA_KS(mf_gs, occ1, occ3, frac_occ)
    mf_es.run(verb=False)
    if not mf_es.converged:
        print("pa-STA-KS excited-state calculation did not converge")
    print("Energy of excited-state singlet (pa-STA-KS):", mf_es.e_tot_oss)
    print("Energy of excited-state triplet (pa-STA-KS):", mf_es.e_tot_t, flush=True)
    mf_es.print_occ_numbers()
    return mf_es


def eval_and_print_maes(uks, sa_, oss_, sta_, ref_):
    """Compute and print mean absolute errors for all four methods.

    Args:
        uks: list of UKS excitation energies in eV.
        sa_: list of pa-SS-KS excitation energies in eV.
        oss_: list of pa-OSS-KS excitation energies in eV.
        sta_: list of pa-STA-KS excitation energies in eV.
        ref_: list of TBE reference values in eV.
    """
    uks = np.array(uks)
    sa_ = np.array(sa_)
    oss_ = np.array(oss_)
    sta_ = np.array(sta_)
    ref_ = np.array(ref_)
    mae = np.mean(np.abs(uks - ref_))
    print(f"MAE (UKS)       = {mae:.2f}")
    mae = np.mean(np.abs(sa_ - ref_))
    print(f"MAE (pa-SS-KS)  = {mae:.2f}")
    mae = np.mean(np.abs(oss_ - ref_))
    print(f"MAE (pa-OSS-KS) = {mae:.2f}")
    mae = np.mean(np.abs(sta_ - ref_))
    print(f"MAE (pa-STA-KS) = {mae:.2f}")


def print_summary(res):
    """Print a summary table of excitation energies and MAEs for all systems.

    Prints three sections — singlet excitations, triplet excitations, and
    singlet-triplet gaps — each followed by MAEs against TBE references.
    Systems without a TBE value for a given multiplicity are skipped.

    Args:
        res: dict mapping system name to a result dict with keys
             "UKS S/T", "pa-SS-KS S/T", "pa-OSS-KS S/T", "pa-STA-KS S/T", "TBE S/T"
             (excitation energies in eV, None if not available).
    """
    print("\nSUMMARY")

    print("\nSINGLET EXCITATIONS:")
    calc, calc_sa, calc_oss, calc_sta, ref = [], [], [], [], []
    for system, _ in res.items():
        if res[system]["TBE S"] is not None:
            calc.append(res[system]["UKS S"])
            calc_sa.append(res[system]["pa-SS-KS S"])
            calc_oss.append(res[system]["pa-OSS-KS S"])
            calc_sta.append(res[system]["pa-STA-KS S"])
            ref.append(res[system]["TBE S"])
            e = res[system]["UKS S"]
            e_sa = res[system]["pa-SS-KS S"]
            e_oss = res[system]["pa-OSS-KS S"]
            e_sta = res[system]["pa-STA-KS S"]
            e_ref = res[system]["TBE S"]
            print(f"{system:30}  {e:.2f}  {e_sa:.2f}  {e_oss:.2f}  {e_sta:.2f}  {e_ref:.2f}")

    eval_and_print_maes(calc, calc_sa, calc_oss, calc_sta, ref)

    print("\nTRIPLET EXCITATIONS:")
    calc, calc_sa, calc_oss, calc_sta, ref = [], [], [], [], []
    for system, _ in res.items():
        if res[system]["TBE T"] is not None:
            calc.append(res[system]["UKS T"])
            calc_sa.append(res[system]["pa-SS-KS T"])
            calc_oss.append(res[system]["pa-OSS-KS T"])
            calc_sta.append(res[system]["pa-STA-KS T"])
            ref.append(res[system]["TBE T"])
            e = res[system]["UKS T"]
            e_sa = res[system]["pa-SS-KS T"]
            e_oss = res[system]["pa-OSS-KS T"]
            e_sta = res[system]["pa-STA-KS T"]
            e_ref = res[system]["TBE T"]
            print(f"{system:30}  {e:.2f}  {e_sa:.2f}  {e_oss:.2f}  {e_sta:.2f}  {e_ref:.2f}")

    eval_and_print_maes(calc, calc_sa, calc_oss, calc_sta, ref)

    print("\nSINGLET-TRIPLET GAPS:")
    calc, calc_sa, calc_oss, calc_sta, ref = [], [], [], [], []
    for system, _ in res.items():
        if res[system]["TBE S"] is not None and res[system]["TBE T"] is not None:
            st = res[system]["UKS S"] - res[system]["UKS T"]
            st_sa = res[system]["pa-SS-KS S"] - res[system]["pa-SS-KS T"]
            st_oss = res[system]["pa-OSS-KS S"] - res[system]["pa-OSS-KS T"]
            st_sta = res[system]["pa-STA-KS S"] - res[system]["pa-STA-KS T"]
            st_ref = res[system]["TBE S"] - res[system]["TBE T"]
            print(f"{system:30}  {st:.2f}  {st_sa:.2f}  {st_oss:.2f}  {st_sta:.2f}  {st_ref:.2f}")
            calc.append(st)
            calc_sa.append(st_sa)
            calc_oss.append(st_oss)
            calc_sta.append(st_sta)
            ref.append(st_ref)

    eval_and_print_maes(calc, calc_sa, calc_oss, calc_sta, ref)


def calc_quest1(config):
    """Run all four DeltaSCF methods on each system in the QUEST1 dataset.

    For each molecule, performs a ground-state RKS calculation, then runs
    UKS, pa-SS-KS, pa-OSS-KS, and pa-STA-KS excited-state calculations for the
    specified HOMO→LUMO excitation. Writes per-system results to results.json
    and prints a summary with MAEs against TBE reference values.

    Args:
        config: dict with keys:
            basis      - basis set name (e.g. "aug-cc-pVQZ")
            auxbasis   - auxiliary basis for density fitting
            xc         - exchange-correlation functional (e.g. "PBE")
            excitation - default [from, to] excitation indices (1-based)
            exceptions - dict mapping system name to a custom excitation index
            grid_level - DFT integration grid level (0-9)
            frac_occ   - if True, use fractional occupations for degenerate orbitals
            systems    - list of system names to run, or None to run all
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
        mol.build()

        # Ground-state singlet
        mf = dft.RKS(mol, xc=config["xc"]).density_fit(auxbasis=config["auxbasis"])
        mf.grids.level = config["grid_level"]
        mf.run()
        if not mf.converged:
            print("RKS ground-state calculation did not converge")
        print("Energy of ground-state singlet:", mf.e_tot, flush=True)
        mf = mf.to_uks()

        exci = config["exceptions"][system] if system in config["exceptions"] else config["excitation"]

        mf_mixed_singlet = run_singlet(mf, config["frac_occ"], exci, UKS)
        mf_triplet = run_triplet(mf, config["frac_occ"], exci, UKS)

        mf_mixed_singlet_sa = run_singlet(mf, config["frac_occ"], exci, pa_SS_KS)
        mf_triplet_sa = run_triplet(mf, config["frac_occ"], exci, pa_SS_KS)

        mf_oss = run_oss(mf, config["frac_occ"], exci)

        mf_sta = run_sta(mf, config["frac_occ"], exci)

        exc_s = (2 * mf_mixed_singlet.e_tot - mf_triplet.e_tot - mf.e_tot) * HA_TO_EV
        exc_t = (mf_triplet.e_tot - mf.e_tot) * HA_TO_EV
        print(f"Singlet Excitation energy (UKS):    {exc_s:.4f} eV")
        print(f"Triplet Excitation energy (UKS):    {exc_t:.4f} eV")
        results[system]["UKS S"] = exc_s
        results[system]["UKS T"] = exc_t

        exc_s = (2 * mf_mixed_singlet_sa.e_tot - mf_triplet_sa.e_tot - mf.e_tot) * HA_TO_EV
        exc_t = (mf_triplet_sa.e_tot - mf.e_tot) * HA_TO_EV
        print(f"Singlet Excitation energy (pa-SS-KS):  {exc_s:.4f} eV")
        print(f"Triplet Excitation energy (pa-SS-KS):  {exc_t:.4f} eV")
        results[system]["pa-SS-KS S"] = exc_s
        results[system]["pa-SS-KS T"] = exc_t

        exc_s = (mf_oss.e_tot - mf.e_tot) * HA_TO_EV
        print(f"Singlet Excitation energy (pa-OSS-KS): {exc_s:.4f} eV")
        results[system]["pa-OSS-KS S"] = exc_s
        results[system]["pa-OSS-KS T"] = exc_t  # from previous triplet pa-SS-KS calculation

        exc_s = (mf_sta.e_tot_oss - mf.e_tot) * HA_TO_EV
        exc_t = (mf_sta.e_tot_t - mf.e_tot) * HA_TO_EV
        print(f"Singlet Excitation energy (pa-STA-KS): {exc_s:.4f} eV")
        print(f"Triplet Excitation energy (pa-STA-KS): {exc_t:.4f} eV")
        results[system]["pa-STA-KS S"] = exc_s
        results[system]["pa-STA-KS T"] = exc_t

        elapsed_mol = fmt_time(time.time() - t_mol)
        elapsed_total = fmt_time(time.time() - t_total)
        print(f"Elapsed: {elapsed_mol}  Total: {elapsed_total}", flush=True)

    with open("results.json", "w", encoding="utf-8") as file_obj:
        print(json.dumps(results, indent=4), file=file_obj)

    print_summary(results)


base_config = {
    "basis": "aug-cc-pVQZ",
    "auxbasis": "aug-cc-pV5Z-RIFIT",
    "xc": "PBE",
    "excitation": [0, 1],
    "exceptions": {"Hydrogen sulfide": [0, 2]},
    "grid_level": 3,
    "frac_occ": False,
    "systems": None,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="quest1", description="quest1 with DFT")
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
