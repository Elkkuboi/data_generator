# synthforest

Paint synthetic forests (tree locations and attributes) and export them as
treemaps in the same format as the laser-scanning-derived treemaps of the
forest recreation research project, so the same analysis code runs on
synthetic and real data.

Work in progress: the engine, command line, tests and the viewer
(`python synthforest.py view forest.rds`) are done; the painting editor follows.

```
pip install -r requirements.txt
python synthforest.py generate examples/clearing_and_rows.json --seed 42 --out forest --formats csv,rds,png
python synthforest.py report forest.csv
python -m unittest discover -s tests
```
