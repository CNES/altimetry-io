# Altimetry_io

Provide a unified way to read a data collection, independent of the underlying data representation.

- Relies on [FCollections](https://robin-cls.github.io/fcollections/index.html#) for netcdf files collections reading
- Can also read Zcollections or CLS tables


## Quick start

```bash
conda install altimetry-io
```

```python
from altimetry.io import AltimetryData, FileCollectionSource

alti_data = AltimetryData(
    source=FileCollectionSource(
        path=output_dir,
        ftype="SWOT_L3_LR_SSH",
        subset="Unsmoothed"
    ),
)

ds = alti_data.query_orbit(
    cycle_number=13,
    pass_number= [153, 155, 157, 181, 183, 209, 211, 237],
    variables=["time", "latitude", "longitude", "quality_flag", "ssha_unedited"],
    polygon=(-151, -109, 71, 78)
)
```

## Documentation

📘 **Full documentation:**
https://robin-cls.github.io/altimetry_io/index.html#

## Project status

⚠️ This project is still subject to breaking changes. Versioning will reflects
the breaking changes using SemVer convention

## License

Apache 2.0 — see [LICENSE](LICENSE)
