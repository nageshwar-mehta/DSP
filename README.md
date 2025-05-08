# Trilateration Python Implementation

This directory contains a pure-Python port of the Julia trilateration project, complete with data and scripts to reproduce results and plots.

## Structure

```
present/python/
├── data/                # Real-world CSV data
│   ├── wifis.csv
│   └── scans.csv
├── requirements.txt     # Python dependencies
├── trilat.py            # Pure-Python trilat and trilat_A functions
├── compare.py           # Simple consistency test against true solution
├── real_data.py         # Runs trilateration on real data and plots results
└── venv/                # (Optional) Python virtual environment
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running on Real Data

This will read `data/wifis.csv` and `data/scans.csv`, perform trilateration, display the plot, and compare metrics with the Julia implementation.

```bash
python real_data.py
```

## Running Unit Tests

A basic consistency check between the two algorithms (`trilat` and `trilat_A`) against a random ground truth:

```bash
python compare.py
```

## Outputs

- `real_data_plot.png`: Generated plot of ground truth vs. estimates.
- Console metrics showing median and mean errors.

Ready to share—everything needed is in this folder.
