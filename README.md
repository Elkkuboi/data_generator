# synthforest

Paint synthetic forests (tree locations and attributes) and export them as
treemaps in the same format as the laser-scanning-derived treemaps of the
forest recreation research project, so the same analysis code runs on
synthetic and real data.

Work in progress: milestone 1 (engine, command line, tests) is done; the
viewer and the painting editor follow.

```
pip install -r requirements.txt
python synthforest.py generate examples/clearing_and_rows.json --seed 42 --out forest --formats csv,rds,png
python synthforest.py report forest.csv
python -m unittest discover -s tests
```
