# 🌊 Hybrid Eddy Detection and Tracking for the Gulf of California (2010–2024)

This repository contains the **Python workflows and notebooks** used to generate the *Gulf of California Mesoscale Eddy Catalog (2010–2024)*.
The complete dataset — including daily detection maps and temporal tracking catalogs — is archived and openly available on **Zenodo**.

📦 **Dataset DOI:** [10.5281/zenodo.17409704](https://doi.org/10.5281/zenodo.17409704)

---

## 🧠 Overview

The workflow implements a **hybrid physical detection algorithm** that combines:

1. The **normalized Okubo–Weiss (OW)** parameter (Okubo, 1970; Weiss, 1991)
2. The **velocity geometry criterion** (Nencioli et al., 2010)
3. The **closed-contour selection** (Chelton et al., 2011)

These steps are applied to daily **detrended SSHA** fields derived from the *NeurOST* dataset (PO.DAAC, 0.1° spatial resolution).
Temporal **tracking** is performed through a cost-based inertial matching algorithm linking coherent eddy detections across days.

---

## 🧬 Repository Structure

```bash
├── notebooks/
│   ├── 01_preprocessing_detrend2D.ipynb     # Spatial detrending (A·lon + B·lat + C)
│   ├── 02_detection_method.ipynb            # Hybrid detection (OW + Nencioli + contours)
│   ├── 03_tracking_method.ipynb             # Inertial cost-based tracking
│   └── figures/                             # Example visualizations of daily detections
│
├── src/
│   ├── eddy_detection_utils.py              # Helper functions for OW and contour logic
│   └── eddy_tracking_utils.py               # Tracking and trajectory matching tools
│
├── README.md                                # (this file)
├── LICENSE                                  # CC-BY 4.0 license
└── CITATION.cff                             # Citation metadata for GitHub / Zenodo
```

---

## 🔗 Data Access

The output catalogs (NetCDF + CSV) are hosted on **Zenodo**:

| File                                                                                 | Description                                                            |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| [`eddy_detections_hybrid_GoC_2010_2024.nc`](https://doi.org/10.5281/zenodo.17409704) | Daily eddy detections (centroid, diameter, polarity, vorticity, etc.)  |
| `tracks_catalog_full.nc` / `.csv`                                                    | Full tracking catalog (971 trajectories, including short-lived eddies) |
| `tracks_catalog_min14d.nc` / `.csv`                                                  | Filtered tracking catalog (394 coherent trajectories, ≥14 detections)  |

🧭 All files follow CF-conventions and standard geophysical units (km, s⁻¹, s⁻²).
The dataset can be directly opened in Python with **xarray**, **netCDF4**, or **pandas**.

---

## 🚀 Quick Start

```python
import xarray as xr
import pandas as pd

# Load the main detection catalog
ds = xr.open_dataset("eddy_detections_hybrid_GoC_2010_2024.nc")
print(ds)

# Load the filtered tracking catalog (CSV version)
df = pd.read_csv("tracks_catalog_min14d.csv")
print(df.head())
```

---

## 🧾 Citation

If you use this code or dataset, please cite:

> Pérez-Corona, Y., Torres, H., & Ramos-Musalem, K. (2025).
> *Eddy Catalog for the Gulf of California (2010–2024) Derived from a Hybrid Detection and Tracking Algorithm.*
> Zenodo. [https://doi.org/10.5281/zenodo.17409704](https://doi.org/10.5281/zenodo.17409704)

---

## 🩶 References

* Okubo, A. (1970). *Horizontal dispersion of floatable particles in the vicinity of velocity singularities.* Deep-Sea Research, 17(3), 445–454.
* Weiss, J. (1991). *The dynamics of enstrophy transfer in two-dimensional hydrodynamics.* Physica D, 48, 273–294.
* Nencioli, F., Dong, C., Dickey, T., Washburn, L., & McWilliams, J. C. (2010). *A vector geometry-based eddy detection algorithm and its application to a high-resolution numerical model product and HF radar surface velocities in the Southern California Bight.* J. Atmos. Ocean. Technol., 27(3), 564–579.
* Chelton, D. B., Schlax, M. G., & Samelson, R. M. (2011). *Global observations of nonlinear mesoscale eddies.* Progress in Oceanography, 91(2), 167–216.

---

## 👩🏻‍🔬 Author

**Yuritzy Pérez-Corona**
Ph.D. Candidate, Physical Oceanography
Centre for Scientific Research and Higher Education of Ensenada (CICESE)
📧 [yuritzy@cicese.edu.mx](mailto:yuritzy@cicese.edu.mx)

**Supervisors:**

* Héctor Torres — Jet Propulsion Laboratory (JPL)
* Karina Ramos-Musalem — CICESE

---

## 📜 License

This repository and its data are released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
You are free to use, share, and adapt the materials with appropriate credit.

---

*Last updated: 2025-10-21*
