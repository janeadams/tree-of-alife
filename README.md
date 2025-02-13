# Tree of (a)Life

## Background

- [Background info](https://itakoyak.wordpress.com/2016/02/20/open-ended-evolution-at-last-some-data/)
- [Videos](https://itakoyak.wordpress.com/2017/02/21/oee-videos/) -- 1st video: some individuals venture off the map where no energy is available, yet survive for several generations and come back to the map. 2nd video: Life emerges once in "water" (bottom) and a different branch emerges in "air" (top). Eventually the waterlife evolves a mutation that helps it invade the air nation. 3rd: New mutants repeatedly extinguish their ancestors.

## Quick start

All settings are configured in `settings.json`, including the data path. Once you have [installed MongoDB Community](https://www.mongodb.com/docs/manual/installation/) and [Compass](https://www.mongodb.com/docs/compass/current/install/), you can run `00_build_database.ipynb` to build the database from your files. Make sure you have run `mongod` from the command line to restart MongoDB after a shutdown.

Once the database is built, you can run `01_build_visualizations.ipynb` to walk through the data collection, transformation, and visualization steps. Note that there is caching built into this system, so you should be able to restart and rerun notebooks without transforming the data from the database all over again. Any time parameters are changed in `settings.json`, the transformation process will rerun from scratch.

New figures will be stored in `tree/` under the parameters for that run. Old figures, including animations, from previous scripts are still in `figs/`. If you are curious how those were generated, you can find older code in `utils/_archive.zip` to save space. The bulk of the code currently used in the workflow is in `utils/utils.py`.

The poster and exported documents are in `poster`. Again, older versions are archived in a zip file at `poster/_archive.zip` to save space.

## Development

`requirements.txt` is generated with pipreqs:

```bash
pipreqs  . --encoding=iso-8859-1 --ignore ".venv" --scan-notebooks
```