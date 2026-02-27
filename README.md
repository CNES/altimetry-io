# Altimetry-io

A library providing a unified way to read an altimetry data collection, independent of the underlying data representation.

- Relies on [Files Collections](https://cnes.github.io/fcollections/) for NetCDF files collections reading
- Can also read ZCollections format

## Installation

```bash
conda install altimetry-io -c conda-forge
```

## Use

```python
from altimetry.io import AltimetryData, FileCollectionSource

alti_data = AltimetryData(
    source=FileCollectionSource(
        # Path to a local directory containing NetCDF data
        path="data_dir",
        # Available ftype: SWOT_L2_LR_SSH, SWOT_L3_LR_SSH, SWOT_L3_LR_WIND_WAVE, NADIR_L2, NADIR_L3
        ftype="SWOT_L3_LR_SSH",
        # Available ftype for SWOT_L3_LR_SSH: Basic, Expert, Unsmoothed, Technical
        subset="Unsmoothed"
    ),
)

ds = alti_data.query_orbit(
    cycle_number=13,
    pass_number= [153, 155],
    variables=["time", "latitude", "longitude", "quality_flag", "ssha_unedited"],
    polygon=(-151, -109, 71, 78)
)
print(ds.sizes)
```

Output:

```text
Frozen({'num_lines': 15893, 'num_pixels': 519})
```

```python
print(list(ds.data_vars))
```

Output:

```text
['quality_flag', 'ssha_unedited', 'cycle_number', 'pass_number']
```

## Project status

⚠️ This project is still subject to breaking changes. Versioning will reflects
the breaking changes using SemVer convention

## License

Apache 2.0 — see [LICENSE](LICENSE)
