# Offline scientific tools

Ordinary `python` / `python3` uses the main scientific Python environment.
Installed versions are listed in `/opt/image-provenance/python-installed.txt`;
the direct package list is `/opt/image-provenance/requirements.in`.

- Symbolic/numerical work: NumPy, SciPy, SymPy, mpmath, gmpy2, python-flint,
  galois and NetworkX.
- Constraints/optimization: Z3 (`import z3`), PySAT (`from pysat.solvers import Solver`),
  CVXPY, CVXOPT, HiGHS and OR-Tools.
- Physics/chemistry: Astropy, QuTiP, Pint, RDKit, PySCF, ASE, pymatgen, Cantera,
  chemicals, thermo and periodictable. PySCF basis sets and Cantera's packaged
  mechanisms are local. External catalogs, databases, pseudopotentials and
  pretrained model weights are not generally bundled.
- Data/statistics/geometry: pandas, statsmodels, scikit-learn, scikit-image,
  matplotlib, h5py, BioPython, Shapely and trimesh. PyTorch and JAX run on CPU.
- Native C/C++/Fortran compilation: gcc, g++, gfortran, make and cmake.

SageMath has its own environment. Use `sage -c 'factor(2^127 - 1)'`, run a
`.sage` file with `sage file.sage`, or use `sage -python file.py` for Python
code that imports `sage.all`. No environment activation is needed.

Lean and its matching Mathlib are installed, including compiled dependencies.
Put your proof in a `.lean` file, then run:

```sh
cd /opt/mathlib
lake env lean /workspace/proof.lean
```

`import Mathlib` works without downloading dependencies. Do not run `lake update`
or create a new project that needs to fetch Mathlib. Ordinary Lean files that
only use the standard library can run with `lean file.lean` anywhere.

Astropy uses bundled IERS data with automatic downloading disabled. Accuracy
checks remain enabled; dates outside supported coverage may fail. Scientific
libraries default to one BLAS/OpenMP thread; adjust their thread variables when
appropriate for the allocated CPU resources. No tools are warmed up at startup.
