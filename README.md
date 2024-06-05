# Tree of (a)Life

## Background

- [Background info](https://itakoyak.wordpress.com/2016/02/20/open-ended-evolution-at-last-some-data/)
- [Videos](https://itakoyak.wordpress.com/2017/02/21/oee-videos/) -- 1st video: some individuals venture off the map where no energy is available, yet survive for several generations and come back to the map. 2nd video: Life emerges once in "water" (bottom) and a different branch emerges in "air" (top). Eventually the waterlife evolves a mutation that helps it invade the air nation. 3rd: New mutants repeatedly extinguish their ancestors.

## Quick start

The expectation is that data exists in `data/SummaryIndividuals.csv`.

## Development

`requirements.txt` is generated with pipreqs:

```bash
pipreqs  . --encoding=iso-8859-1 --ignore ".venv" --scan-notebooks
```