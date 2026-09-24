# # Prepare clinical MRI cohort for direct Jacobian vs Wasserstein
# 
# Build a lean, session-level clinical cohort. MRI sessions are matched to the nearest diagnosis and CDR visits from the same participant.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = next(
    folder for folder in (Path.cwd(), *Path.cwd().parents)
    if (folder / 'data' / 'database_finale_labels_corrette.csv').is_file()
)

BASE_PATH = PROJECT_ROOT / 'data' / 'database_finale_labels_corrette.csv'
DIAGNOSIS_PATH = PROJECT_ROOT / 'csv_source_oasis' / 'OASIS3_UDSd1_diagnoses.csv'
CDR_PATH = PROJECT_ROOT / 'csv_source_oasis' / 'OASIS3_UDSb4_cdr.csv'
RUN_ROOT = PROJECT_ROOT / 'outputs' / 'cohorts' / 'oasis3' / 'runs' / 'connected-atlas-8cf3dc8f'
DIRECT_ROOT = RUN_ROOT / 'parcel_vectors'
WEIGHTED_DEGREE_ROOT = RUN_ROOT / 'graphs' / 'unthresholded' / 'was' / 'expW'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'derived' / 'oasis3_training_cohort.csv'


MAX_CLINICAL_GAP_DAYS = 180

# 1. Read only identifiers needed for clinical matching.
base_columns = ['OASISID', 'Enr-Day']
mri_sessions = pd.read_csv(BASE_PATH, usecols=base_columns, dtype={'Enr-Day': 'string'})
mri_sessions['mri_day'] = (
    mri_sessions['Enr-Day'].str.extract(r'(\d+)')[0].astype('Int64')
)
mri_sessions['mri_row'] = np.arange(len(mri_sessions))

# 2. Retain only diagnosis columns needed to define a clinical group.
diagnosis_columns = [
    'OASISID', 'days_to_visit', 'NORMCOG', 'DEMENTED',
    'PROBAD', 'PROBADIF', 'alzdis', 'alzdisif'
]
diagnosis_visits = pd.read_csv(DIAGNOSIS_PATH, usecols=diagnosis_columns).copy()
diagnosis_visits['d1_visit_day'] = pd.to_numeric(
    diagnosis_visits.pop('days_to_visit'), errors='coerce'
)
diagnosis_visits = diagnosis_visits.rename(columns={
    column: f'd1_{column.lower()}'
    for column in diagnosis_visits.columns
    if column not in {'OASISID', 'd1_visit_day'}
})

# 3. Retain CDR severity and cognitive-status columns.
cdr_columns = ['OASISID', 'days_to_visit', 'CDRTOT']
cdr_visits = pd.read_csv(CDR_PATH, usecols=cdr_columns).copy()
cdr_visits['cdr_visit_day'] = pd.to_numeric(
    cdr_visits.pop('days_to_visit'), errors='coerce'
)
cdr_visits = cdr_visits.rename(columns={
    'CDRTOT': 'cdr_total',
})

# 4. Match each MRI to the nearest visit of one clinical table, within the same OASISID.
def nearest_visit_match(mri, visits, visit_day_column, prefix):
    matches = []
    value_columns = [column for column in visits.columns if column != 'OASISID']

    for oasis_id, scans in mri.groupby('OASISID', sort=False):
        left = (
            scans[['mri_row', 'mri_day']].dropna()
            .assign(mri_day=lambda frame: frame['mri_day'].astype('int64'))
            .sort_values('mri_day')
        )
        right = visits.loc[visits['OASISID'].eq(oasis_id), value_columns].dropna(
            subset=[visit_day_column]
        ).assign(**{visit_day_column: lambda frame: frame[visit_day_column].astype('int64')})
        right = right.sort_values(visit_day_column)

        if right.empty:
            empty = left.copy()
            for column in value_columns:
                empty[column] = np.nan
            matches.append(empty)
        else:
            matches.append(pd.merge_asof(
                left, right, left_on='mri_day', right_on=visit_day_column, direction='nearest'
            ))

    matched = pd.concat(matches, ignore_index=True)
    matched[f'{prefix}_gap_days_signed'] = matched[visit_day_column] - matched['mri_day']
    matched[f'{prefix}_gap_days_abs'] = matched[f'{prefix}_gap_days_signed'].abs()
    return matched.drop(columns='mri_day')

d1_matches = nearest_visit_match(mri_sessions, diagnosis_visits, 'd1_visit_day', 'd1')
cdr_matches = nearest_visit_match(mri_sessions, cdr_visits, 'cdr_visit_day', 'cdr')

clinical_mri = (
    mri_sessions
    .merge(d1_matches, on='mri_row', how='left', validate='one_to_one')
    .merge(cdr_matches, on='mri_row', how='left', validate='one_to_one')
)

# 5. Apply the time window, then create conservative clinical labels.
clinical_mri['d1_within_window'] = clinical_mri['d1_gap_days_abs'].le(MAX_CLINICAL_GAP_DAYS)
clinical_mri['cdr_within_window'] = clinical_mri['cdr_gap_days_abs'].le(MAX_CLINICAL_GAP_DAYS)

primary_ad = (
    clinical_mri['d1_demented'].eq(1)
    & (
        (clinical_mri['d1_alzdis'].eq(1) & clinical_mri['d1_alzdisif'].eq(1))
        | (clinical_mri['d1_probad'].eq(1) & clinical_mri['d1_probadif'].eq(1))
    )
)
clinically_normal = clinical_mri['d1_normcog'].eq(1) & clinical_mri['cdr_total'].eq(0)

clinical_mri['clinical_group'] = np.select(
    [
        primary_ad & clinical_mri['d1_within_window'] & clinical_mri['cdr_within_window'],
        clinically_normal & clinical_mri['d1_within_window'] & clinical_mri['cdr_within_window'],
    ],
    ['primary_ad_dementia', 'clinically_normal'],
    default=pd.NA,
)
clinical_mri['clinical_label_ad'] = clinical_mri['clinical_group'].map({
    'clinically_normal': 0,
    'primary_ad_dementia': 1,
}).astype('Int64')


# 6. Record whether both precomputed imaging feature vectors are available.
clinical_mri['subject_id'] = (
    'sub-' + clinical_mri['OASISID'].str.removeprefix('OAS3')
)
clinical_mri['direct_mean_path'] = clinical_mri['subject_id'].map(
    lambda subject: str(DIRECT_ROOT / subject / 'direct_mean.dat')
)
clinical_mri['weighted_degree_path'] = clinical_mri['subject_id'].map(
    lambda subject: str(WEIGHTED_DEGREE_ROOT / subject / 'weighted_degree.dat')
)
clinical_mri['has_direct_mean'] = clinical_mri['direct_mean_path'].map(
    lambda path: Path(path).is_file()
)
clinical_mri['has_weighted_degree'] = clinical_mri['weighted_degree_path'].map(
    lambda path: Path(path).is_file()
)

model_cohort = clinical_mri.loc[
    clinical_mri['clinical_group'].notna()
    & clinical_mri['has_direct_mean']
    & clinical_mri['has_weighted_degree']
].copy()


# 7. Save the lean modelling table. No HStatus-derived label is retained.
output_columns = [
    'subject_id', 'Enr-Day', 'clinical_label_ad',
    'direct_mean_path', 'weighted_degree_path',
]

model_cohort = model_cohort[output_columns].copy()
if model_cohort.empty:
    raise ValueError('No labelled subjects have both imaging vectors.')
if model_cohort['subject_id'].duplicated().any():
    raise ValueError('Repeated subjects require grouped CV or a single-session cohort.')
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
model_cohort.sort_values(['clinical_label_ad', 'subject_id']).to_csv(
    OUTPUT_PATH, index=False
)

print(f'Saved: {OUTPUT_PATH}')
del clinical_mri, mri_sessions, diagnosis_visits, cdr_visits, d1_matches, cdr_matches

# ## Training the ML Model -- Testing ROC-AUC 

# One vector per subject, preserving model_cohort order
direct_mean_rows = []

for path in model_cohort["direct_mean_path"]:
    values = np.fromfile(path, dtype=np.float64)
    direct_mean_rows.append(values)

# Shape: n_subjects × 83_442
X_direct = np.vstack(direct_mean_rows)
del direct_mean_rows

# Labels aligned with the same row order
y = model_cohort["clinical_label_ad"].to_numpy(dtype=np.int64)

print(X_direct.shape)
print(y.shape)


# One Wasserstein weighted-degree vector per subject,
# preserving the same model_cohort row order
weighted_degree_rows = []

for path in model_cohort["weighted_degree_path"]:
    values = np.fromfile(path, dtype=np.float64)
    weighted_degree_rows.append(values)

# Shape: n_subjects × 83_442
X_weighted_degree = np.vstack(weighted_degree_rows)
del weighted_degree_rows

# Same labels, aligned with the same subject order.

print(X_weighted_degree.shape)
print(y.shape)

X_combined = np.hstack([X_direct, X_weighted_degree])

print(X_combined.shape)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


def evaluate_auc(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.01,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )),
    ])

    # Every score comes from a fold where that subject was not used for training.
    oof_probability = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    return roc_auc_score(y, oof_probability), oof_probability


auc_direct, pred_direct = evaluate_auc(X_direct, y)
auc_weighted, pred_weighted = evaluate_auc(X_weighted_degree, y)
auc_combined, pred_combined = evaluate_auc(X_combined, y)

print(f"Direct Jacobian:  {auc_direct:.3f}")
print(f"Weighted degree:  {auc_weighted:.3f}")
print(f"Combined:         {auc_combined:.3f}")
