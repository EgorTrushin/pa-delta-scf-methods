# Potential-averaged and optimized effective potential ΔSCF methods

This repository contains the implementation of the methods introduced in:

> E. Trushin, O. Bertleff, A. Görling. Potential-Averaged ΔSCF Methods: Underlying Formalism and Evaluation of Accuracy of Excitation Energies. J. Chem. Theory Comput. **2026**, *22*, 8324–8349. https://doi.org/10.1021/acs.jctc.6c00983

## Published version and ongoing development

The version of the code associated with the published article is permanently archived on Zenodo: https://doi.org/10.5281/zenodo.20270737. The `main` branch of this repository may differ from the version used for the publication. For reproduction of the results reported in the article, please use the archived Zenodo version. The current `main` branch contains subsequent fixes, improvements, and further developments of the methods.

## Methods

All methods are implemented on top of [PySCF](https://pyscf.org).

- **UKS-KS**: standard spin-unrestricted Kohn-Sham method

#### Potential-averaged Kohn-Sham methods

- **pa-SS-KS**: potential-averaged spin-symmetrized Kohn-Sham method
- **pa-OSS-KS**: potential-averaged open-shell singlet Kohn-Sham method
- **pa-STA-KS**: potential-averaged state-averaged Kohn-Sham method

#### Optimized effective potential methods

- **UKS-OEP**: spin-unrestricted Kohn-Sham optimized effective potential method
- **SS-KS-OEP** spin-symmetrized Kohn-Sham optimized effective potential method
- **OSS-KS-OEP** open-shell singlet Kohn-Sham optimized effective potential method
- **STA-KS-OEP** state-averaged Kohn-Sham optimized effective potential method

More about OEP methods and their usage can be learned at https://github.com/EgorTrushin/PyOEP

Each OEP method has two implementations based on orbital swapping and occupation numbers. In orbital swapping case, after each SCF iteration the orbitals are reordered so that the MOM-selected configuration occupies canonical positions. In fractional occupation numbers case, MOM is applied through occupation numbers that track the desired configuration throughout the SCF.

## Repository structure

```
methods/                  # Standard KS methods
  UKS.py                  # Spin-unrestricted KS (UKS)
  pa_SS_KS.py             # Potential-averaged spin-symmetrized KS (pa-SS-KS)
  pa_OSS_KS.py            # Potential-averaged open-shell singlet KS (pa-OSS-KS)
  pa_STA_KS.py            # Potential-averaged state-averaged KS (pa-STA-KS)
  mom.py                  # Maximum overlap method

methods_oep/              # OEP variants
  dftoep.py               # Ground-state KS-OEP
  osdftoep_swap.py        # UKS-OEP and SS-KS-OEP with orbital swapping
  osdftoep_occ.py         # UKS-OEP and SS-KS-OEP with occupation numbers
  osdftoep_oss_swap.py    # OSS-KS-OEP with orbital swapping
  osdftoep_oss_occ.py     # OSS-KS-OEP with occupation numbers
  osdftoep_sta_swap.py    # STA-KS-OEP with orbital swapping
  osdftoep_sta_occ.py     # STA-KS-OEP with occupation numbers

tests/                    # Test suite
  test_h2o.py             # Standard KS tests on H₂O
  test_n2.py              # Standard KS tests on N₂
  test_h2o_oep_swap.py    # OEP with orbital swapping, tests on H₂O
  test_n2_oep_swap.py     # OEP with orbital swapping, tests on N₂
  test_h2o_oep_occ.py     # OEP with occupation numbers, tests on H₂O
  test_n2_oep_occ.py      # OEP with occupation numbers, tests on N₂

sets/                     # Molecular geometries and reference energies
  quest1_lowest.py        # QUEST1 lowest excited states
  quest1_higher.py        # QUEST1 higher excited states
  quest6.py               # QUEST6 dataset

configs/                  # Calculation configuration files (YAML)

results/                  # Calculation outputs and parsed results
  quest1_lowest/          # QUEST1 lowest excited states
  quest1_higher/          # QUEST1 higher excited states
  quest6/                 # QUEST6 dataset

csv/                      # Excitation energy and MAE tables (CSV)

examples.py               # Standard KS examples: H₂O and N₂
examples_oep.py           # OEP examples: H₂O and N₂
pa-delta-scf-methods.ipynb      # Analysis notebook
supplementary-material.ipynb    # Supplementary material notebook
graphical_abstracts.py          # Script to generate the graphical abstract
```

## Installation

Clone the repository and install the required packages:

```bash
pip install pyscf basis_set_exchange
```

NumPy and SciPy are included with PySCF. For running tests, `pytest` is also required:

```bash
pip install pytest
```

To make the `methods/` and `utils/` packages importable, either run scripts from the project root or add the project root to `PYTHONPATH`.

## Examples

A full comparison of all four standard methods on H₂O and N₂ is available in
`examples.py`, and OEP examples (both `_swap` and `_occ` implementations) are in
`examples_oep.py`:

```bash
python examples.py
python examples_oep.py
```

The example below shows the essential workflow for computing the first singlet
and triplet excitation energies of H₂O with the UKS method:

```python
from pyscf import dft, gto
from methods.UKS import UKS

# Ground-state calculation
mol = gto.M(atom="O 0.0 0.0 -0.06990256; H 0.0 0.75753241 0.51843495; H 0.0 -0.75753241 0.51843495",
            basis="aug-cc-pVTZ")
mol.verbose = 0
mf = dft.RKS(mol, xc="PBE").density_fit(auxbasis="aug-cc-pV5Z-RIFIT").run()
mf = mf.to_uks()

# Occupation numbers for HOMO -> LUMO excitation
occ_s = mf.mo_occ.copy()
occ_s[0][mf.nelec[0] - 1] = 0  # remove alpha electron from HOMO
occ_s[0][mf.nelec[0]]     = 1  # add alpha electron to LUMO

occ_t = mf.mo_occ.copy()
occ_t[1][mf.nelec[1] - 1] = 0  # remove beta electron from HOMO
occ_t[0][mf.nelec[0]]     = 1  # add alpha electron to LUMO

# Excited-state calculations
mf_s = UKS(mf, occ_s, frac_occ=False)
mf_s.run(verb=False)

mf_t = UKS(mf, occ_t, frac_occ=False)
mf_t.run(verb=False)

# Excitation energies (Hartree -> eV)
HA_TO_EV = 27.2114
exc_s = (2 * mf_s.e_tot - mf_t.e_tot - mf.e_tot) * HA_TO_EV
exc_t = (mf_t.e_tot - mf.e_tot) * HA_TO_EV
print(f"S1 = {exc_s:.4f} eV")
print(f"T1 = {exc_t:.4f} eV")
```

The STA-KS method optimizes a state-averaged potential for both singlet and
triplet simultaneously, extracting both excitation energies in a single SCF:

```python
from pyscf import dft, gto
from methods.pa_STA_KS import pa_STA_KS

# Ground-state calculation
mol = gto.M(atom="O 0.0 0.0 -0.06990256; H 0.0 0.75753241 0.51843495; H 0.0 -0.75753241 0.51843495",
            basis="aug-cc-pVTZ")
mol.verbose = 0
mf = dft.RKS(mol, xc="PBE").density_fit(auxbasis="aug-cc-pV5Z-RIFIT").run()
mf = mf.to_uks()

# Occupation numbers for HOMO -> LUMO excitation
occ_s = mf.mo_occ.copy()
occ_s[0][mf.nelec[0] - 1] = 0  # remove alpha electron from HOMO
occ_s[0][mf.nelec[0]]     = 1  # add alpha electron to LUMO

occ_t = mf.mo_occ.copy()
occ_t[1][mf.nelec[1] - 1] = 0  # remove beta electron from HOMO
occ_t[0][mf.nelec[0]]     = 1  # add alpha electron to LUMO

# Excited-state calculation
mf_sta = pa_STA_KS(mf, occ_s, occ_t, frac_occ=False)
mf_sta.run(verb=False)

# Excitation energies (Hartree -> eV)
HA_TO_EV = 27.2114
exc_s = (mf_sta.e_tot_oss - mf.e_tot) * HA_TO_EV
exc_t = (mf_sta.e_tot_t - mf.e_tot) * HA_TO_EV
print(f"S1 = {exc_s:.4f} eV")
print(f"T1 = {exc_t:.4f} eV")
```

`pa_SS_KS` and `pa_OSS_KS` follow the same patterns as `UKS` and `pa_STA_KS`
respectively. See `examples.py` for a full comparison of all four methods.

### OEP examples

OEP calculations require a ground-state KS-OEP step first. The example below
shows the UKS-OEP workflow for H₂O using the orbital-swap implementation:

```python
from pyscf import dft, gto
from methods_oep.dftoep import DFTOEP
from methods_oep.osdftoep_swap import OSDFTOEP_swap

mol = gto.M(atom="O 0.0 0.0 -0.06990256; H 0.0 0.75753241 0.51843495; H 0.0 -0.75753241 0.51843495",
            basis="aug-cc-pVTZ")
mol.verbose = 0
mf = dft.RKS(mol, xc="PBE").density_fit(auxbasis="aug-cc-pV5Z-RIFIT").run()

# Ground-state KS-OEP calculation
mf_oep_gs = DFTOEP(mf, "aug-cc-pVDZ-RIFIT")
mf_oep_gs.run(maxit=50, thr_fai_oep=0.05)
e_gs = mf_oep_gs.e_tot

mf = mf.to_uks()

# Occupation numbers for HOMO -> LUMO excitation
occ_s = mf.mo_occ.copy()
occ_s[0][mf.nelec[0] - 1] = 0  # remove alpha electron from HOMO
occ_s[0][mf.nelec[0]]     = 1  # add alpha electron to LUMO

occ_t = mf.mo_occ.copy()
occ_t[1][mf.nelec[1] - 1] = 0  # remove beta electron from HOMO
occ_t[0][mf.nelec[0]]     = 1  # add alpha electron to LUMO

# Excited-state OEP calculations (spin_sym=False for UKS-OEP)
mf_s = OSDFTOEP_swap(mf, "aug-cc-pVDZ-RIFIT", occ_s, spin_sym=False)
mf_s.run(maxit=50, thr_fai_oep=0.05)

mf_t = OSDFTOEP_swap(mf, "aug-cc-pVDZ-RIFIT", occ_t, spin_sym=False)
mf_t.run(maxit=50, thr_fai_oep=0.05)

# Excitation energies (Hartree -> eV)
HA_TO_EV = 27.2114
exc_s = (2 * mf_s.e_tot - mf_t.e_tot - e_gs) * HA_TO_EV
exc_t = (mf_t.e_tot - e_gs) * HA_TO_EV
print(f"S1 = {exc_s:.4f} eV")
print(f"T1 = {exc_t:.4f} eV")
```

The STA-KS-OEP method extracts both excitation energies in a single SCF:

```python
from pyscf import dft, gto
from methods_oep.dftoep import DFTOEP
from methods_oep.osdftoep_sta_swap import OSDFTOEP_STA_swap

mol = gto.M(atom="O 0.0 0.0 -0.06990256; H 0.0 0.75753241 0.51843495; H 0.0 -0.75753241 0.51843495",
            basis="aug-cc-pVTZ")
mol.verbose = 0
mf = dft.RKS(mol, xc="PBE").density_fit(auxbasis="aug-cc-pV5Z-RIFIT").run()

# Ground-state KS-OEP calculation
mf_oep_gs = DFTOEP(mf, "aug-cc-pVDZ-RIFIT")
mf_oep_gs.run(maxit=50, thr_fai_oep=0.05)
e_gs = mf_oep_gs.e_tot

mf = mf.to_uks()

# Occupation numbers for HOMO -> LUMO excitation
occ_s = mf.mo_occ.copy()
occ_s[0][mf.nelec[0] - 1] = 0  # remove alpha electron from HOMO
occ_s[0][mf.nelec[0]]     = 1  # add alpha electron to LUMO

occ_t = mf.mo_occ.copy()
occ_t[1][mf.nelec[1] - 1] = 0  # remove beta electron from HOMO
occ_t[0][mf.nelec[0]]     = 1  # add alpha electron to LUMO

# Excited-state calculation (singlet and triplet in one SCF)
mf_sta = OSDFTOEP_STA_swap(mf, "aug-cc-pVDZ-RIFIT", occ_s, occ_t)
mf_sta.run(maxit=50, thr_fai_oep=0.05)

# Excitation energies (Hartree -> eV)
HA_TO_EV = 27.2114
exc_s = (mf_sta.e_tot  - e_gs) * HA_TO_EV
exc_t = (mf_sta.e_tot3 - e_gs) * HA_TO_EV
print(f"S1 = {exc_s:.4f} eV")
print(f"T1 = {exc_t:.4f} eV")
```

See `examples_oep.py` for a full comparison of all OEP methods on H₂O and N₂,
including the occupation-number implementation.

## Running tests

Run all tests from the project root:

```bash
python -m pytest tests/
```

Run a specific test file:

```bash
# Standard KS methods
python -m pytest tests/test_h2o.py
python -m pytest tests/test_n2.py

# OEP methods — orbital-swap implementation
python -m pytest tests/test_h2o_oep_swap.py
python -m pytest tests/test_n2_oep_swap.py

# OEP methods — occupation-number implementation
python -m pytest tests/test_h2o_oep_occ.py
python -m pytest tests/test_n2_oep_occ.py
```

The `-m` flag is required to ensure the `methods/` and `methods_oep/` packages are importable from the project root. Alternatively, add the project root to `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/pa-delta-scf-methods:$PYTHONPATH"
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
