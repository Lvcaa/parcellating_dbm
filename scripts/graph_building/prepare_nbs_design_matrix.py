"""Write the NBS design matrix in the saved similarity-matrix subject order."""

import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / 'outputs/nbs'


def main():
    with (PROJECT_ROOT / 'data/database_finale_labels_corrette.csv').open(newline='') as file:
        subjects = {row['OASISID'][4:]: row for row in csv.DictReader(file)}
    matrix_metadata = json.loads((OUTPUT_DIR / 'similarities_was_expW.json').read_text())
    subject_order = matrix_metadata['subjects_in_matrix_order']
    rows = [subjects[subject[4:]] for subject in subject_order]
    design = np.array([
        [1, row['HStatus'] == 'Unhealthy', float(row['age at visit']),
         row['GENDER'] == 'Male', float(row['EDUC'])]
        for row in rows
    ], dtype=float)
    output = OUTPUT_DIR / 'design_matrix.txt'
    np.savetxt(output, design, fmt='%.10g')
    output.with_suffix('.json').write_text(json.dumps({
        'columns': ['intercept', 'unhealthy', 'age', 'male', 'education'],
        'group_coding': {'Healthy': 0, 'Unhealthy': 1},
        'sex_coding': {'Female': 0, 'Male': 1},
        'subjects_in_row_order': subject_order,
        'contrast_unhealthy_greater_than_healthy': [0, 1, 0, 0, 0],
        'contrast_healthy_greater_than_unhealthy': [0, -1, 0, 0, 0],
    }, indent=2) + '\n')
    print(f'Saved {output}: {design.shape[0]} rows, {design.shape[1]} columns')


if __name__ == '__main__':
    main()
