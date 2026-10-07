Carolina B. Prado - 20260528


Week 2: 16/Sept Practical Class

  Tested out "decision_tree" and "logistic_regression"
  
  - Logistic Regression is better as the Decision Tree model overfits the model a lot
  
  
  ex: logistic regression results
      
      
      Train accuracy: 0.680
      Test accuracy:  0.678
      Gap (train - test): +0.002 
    
    
    decision tree results
      Train accuracy: 0.829
      Test accuracy:  0.629
      Gap (train - test): +0.199


Week 3: 23/Sept Practical Class

  Learned how to run a diagnosis in our dataset and pre process (cleaning the data)
  ex on the results:
  
    logistic_regression results
      Train accuracy: 0.676
      Test accuracy:  0.657
      Gap (train - test): +0.019
      
    decision tree results
      Train accuracy: 0.792
      Test accuracy:  0.611
      Gap (train - test): +0.180
      
  With the preprocessing, the decision tree is still overfitting the model, and the logistic regression results is generalizing very well.

Week 4: 30/Sept Practical Class


  Changed the evaluation to 5-fold cross-validation and kept a locked test set apart (20%, not evaluated yet)
  Chose target encoder + robust scaler for the preprocessing 
  Added a fairness check: false positive rate (FPR) by race, compared to COMPAS's own score
  ex on the results:

  
    logistic_regression results (5-fold CV)
      Train accuracy: 0.675
      Validation accuracy: 0.672 (std 0.013)
      Gap (train - validation): +0.003

      
    false positive rate by race (people who did not reoffend but were predicted to)
      African-American   my model: 0.26   COMPAS: 0.45
      Caucasian          my model: 0.13   COMPAS: 0.23
      Hispanic           my model: 0.15   COMPAS: 0.23

      
  The logistic regression is generalizing very well (train and validation almost the same).
  My model makes fewer false positives than COMPAS, but African-American defendants are still wrongly flagged about 2x more than Caucasian defendants, even without race as a feature.


Week 5: 07/Oct Practical Class

  Added hyperparameter tuning with Optuna (TPE sampler), on the development set only, every candidate scored by 5-fold CV of the whole pipeline
  Added nested cross-validation to get an honest score for the tuned model (the best trial's score is optimistic)
  The fairness check now also shows people with unknown race ("unknown", 66 people) instead of silently dropping them
  ex on the results:

    decision tree (nested CV, tuned)
      Default: 0.611 (std 0.015), gap +0.084
      Tuned:   0.675 (std 0.020), gap +0.009
      Chosen: max_depth 9, min_samples_leaf 101, gini

  Tuning fixed the decision tree's overfitting: forcing every leaf to hold ~100 people stops it from memorising individuals.
  After tuning, decision tree, random forest and logistic regression are all tied (0.673-0.678, within one std).




# Baseline Predictive Pipeline -- ETAI

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                  # entry point: run the whole pipeline
├── config.yaml               # all tunable settings live here
├── requirements.txt
├── src/
│   ├── bin/
│   │   └── data_diagnostics.py  # EDA-only tools (week 3) -- not used by main.py
│   ├── data.py               # loading
│   ├── preprocessing.py      # leak-safe cleaning, deployable preprocessing pipeline, and train/test split (week 3 grew this file's job well beyond just the split -- same file, same name as week 2)
│   ├── model.py               # model construction
│   ├── evaluate.py           # accuracy  + fairness check
│   └── results.py            # saves each run's report to disk
├── results/                  # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md              # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check) and `src/preprocessing.py` (leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, **and** the train/test split itself, all in the one file rather than split across two) replace the old naive `dropna()`/`pd.get_dummies()` preprocessing; encoder/scaler pair (count encoding + robust scaling) chosen by an empirical grid over 15 repeated splits, checked against the runner-up with a paired comparison so the win isn't just noise; three redundant columns (found via correlation + VIF) dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections -- see "Preprocessing decisions" below. |
| 4 | Cross-validation -- evaluating a model honestly | Locked final test set (20%, stratified, seed 42, never evaluated) via `split_dev_test()`; stratified 5-fold CV of the **whole** pipeline (preprocessing inside every fold) on the development set, reporting train/validation/gap per fold with mean ± std (`src/evaluate.py`); classification report and fairness check now computed on out-of-fold predictions; final model refit on all development rows; `clean_dataset()` made row-preserving with de-duplication moved to the training-only `drop_duplicate_rows()`; sklearn's `TargetEncoder` (internal cross-fitting) replaces `category_encoders`'; target + robust scaling chosen by hand; `src/model.py` gains `dummy` and `random_forest`; `config.yaml` gains `test_set` and `cv` sections; `data_diagnostics.py` moved to `src/bin/` (EDA-only, not used by `main.py`). |
| 5 | Hyperparameter tuning | New `src/tuning.py`: `tune_pipeline()` (Optuna study with a seeded TPE sampler, every trial = 5-fold CV of the whole pipeline on the development set), `nested_cross_validate()` (honest estimate of the tuning procedure: inner folds choose, outer folds judge) and `tuning_report()` (best trials + optimism check); `build_pipeline()` in `src/model.py` is now the single place where preprocessing + model are assembled; `config.yaml` gains a `tuning` section with one search space per model; `main.py` runs nested CV, then tunes on all development rows and refits the tuned pipeline; the fairness report shows rows with unknown race as "unknown"; `optuna` added to `requirements.txt`. |

## Preprocessing decisions

*(New this week -- written straight from the diagnosis in `Practical/W3/notebooks/01_eda_introduction.ipynb` and the empirical grid in `02_preprocessing.ipynb`. Full walkthrough lives in those two notebooks; this is the summary.)*

| Column(s) | Issue found | Mechanism | What was done |
|---|---|---|---|
| `age` | 2.0% missing | MCAR | median impute, no indicator needed |
| `juv_fel_count` | 3.0% missing | MCAR | median impute, no indicator needed |
| `priors_count` | ~7% missing (incl. placeholder tokens) | MNAR -- tied to `age_cat` | median impute + `priors_count_was_missing` flag |
| `c_charge_degree` | 3.2% missing | MNAR -- tied to `age_cat` | mode impute + `c_charge_degree_was_missing` flag |
| `race` | ~1% missing (placeholder tokens) | MCAR | mode impute, no indicator (excluded from model features anyway) |
| `sex` | ~1.5% missing (incl. placeholder tokens) | MCAR | mode impute, no indicator needed |
| `age`, `decile_score`, `juv_fel_count`, `priors_count` | invalid values (out-of-range or negative) | domain rule | converted to `NaN` before imputation |
| `sex` / `race` / `c_charge_degree` / `score_text` | inconsistent category spelling (casing, whitespace, abbreviations) | data entry | canonicalized to one spelling per category |
| whole rows | 72 exact-duplicate rows, all sharing a repeated `id` | data entry | dropped, kept first occurrence |
| `prior_offenses`, `age_in_months`, `juvenile_total` | redundant with other columns (correlation r=1.00, or -- for `juvenile_total` -- an exact sum caught only by VIF) | multicollinearity | dropped |

**Encoder/scaler pair**: target encoding + robust scaling, chosen by hand (week 4): target encoding is compact (one column per feature) and informative; robust scaling uses median/IQR so the few extreme counts don't set the scale.


## Model evaluation

The final test set (20% of the data, stratified, seed 42) is locked: it is never used to fit, compare or choose anything. Every model is evaluated with stratified 5-fold cross-validation on the remaining 80% (the development set, 5,771 rows). The whole pipeline -- imputation, encoding, scaling and model -- is re-fit inside every fold, so a validation fold never influences its own preprocessing. All models use the same folds (`cv.random_state: 42`), so the comparison is like-for-like.

| Model | Holdout accuracy (W3) | CV accuracy (mean ± std) | CV train–val gap |
|---|---|---|---|
| Dummy | 0.550 | 0.549 ± 0.000 | +0.000 |
| Logistic regression | 0.657 | 0.672 ± 0.013 | +0.003 |
| Decision tree | 0.611 | 0.611 ± 0.015 | +0.084 |
| Random forest | — | 0.652 ± 0.017 | +0.080 |

I trust the CV numbers more than the week 3 holdout: they use every development row for validation and come with a standard deviation (~1.5 points), which shows how much a single split can move the score. The week 2/3 conclusion still holds: logistic regression is the best model, about 2 points ahead of the random forest (more than one std), and it doesn't overfit (gap ≈ 0). Both tree-based models overfit by ~8 points with default hyperparameters, which is what week 5's tuning will address. The decision tree's gap dropped from 0.180 (W3) to 0.084, most likely because sklearn's `TargetEncoder` cross-fits its encoding instead of leaking the target into the training rows. All models beat the dummy baseline (0.549).


### Hyperparameter tuning

Hyperparameters are chosen with Optuna (TPE sampler, seed 42) on the development set only: each trial sets one candidate on a fresh copy of the whole pipeline and scores it by stratified 5-fold CV, so imputation, encoding and scaling are re-fit inside every fold of every trial, and every trial uses the same folds. The locked test set is never touched. Because the best trial's score is the maximum of many noisy scores, it is optimistic: the number reported is from **nested cross-validation** -- for each of the 5 outer folds, the whole tuning is run on the other 4 and the winner is scored on the held-out fold. The hyperparameters kept are from the same tuning run once on all development rows. Decision tree and logistic regression: 30 trials; random forest: 15 trials (each trial is ~100× slower).

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Decision tree | 0.611 ± 0.015 | 0.675 ± 0.020 | 0.680 | −0.000 | `max_depth=9`, `min_samples_leaf=101`, `criterion=gini` |
| Logistic regression | 0.672 ± 0.013 | 0.673 ± 0.016 | 0.678 | +0.002 | `C=0.072` |
| Random forest | 0.652 ± 0.017 | 0.678 ± 0.013 | 0.682 | +0.003 | `n_estimators=151`, `max_depth=8`, `min_samples_leaf=26`, `max_features=log2` |

The tree-based models gained the most from tuning (+6.4 points for the decision tree, +2.6 for the random forest, both well above the fold-to-fold std), and their overfitting disappeared (train–validation gap from ~0.08 to ~0.01): large minimum leaf sizes stop them from memorising individual people. Logistic regression gained nothing -- a linear model on a few features with ~5,800 rows has little variance for regularisation to remove. Tuning changed the ranking: all three models are now within half a point of each other, less than one standard deviation, so no model is clearly best on accuracy. I would report the nested CV score (e.g. 0.673 ± 0.016 for logistic regression), not the best trial's score, since only the nested one comes from rows that took no part in choosing the hyperparameters. With accuracy tied, logistic regression remains my choice: it has the lowest false positive rates in every group (African-American 0.25, Caucasian 0.12) and its coefficients are directly interpretable -- though, like every model here, it still wrongly flags African-American defendants about twice as often as Caucasian ones.



## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Environment Troubleshooting

Two Windows issues come up often enough to note here -- if you hit either, this saves you re-diagnosing it from scratch.

**PowerShell blocks the venv activation script, every new terminal.** The `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` line above only fixes it for that one terminal window -- close it and it's back. For a fix that actually sticks across sessions, run this **once**, instead:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
If it still doesn't stick (common on locked-down school/lab machines with a Group Policy that resets it on every logon), skip PowerShell entirely: use **Git Bash** (`source venv/Scripts/activate`) or **cmd.exe** (`venv\Scripts\activate.bat`) instead -- neither is affected by PowerShell's execution policy.

**Windows blocks the terminal/Python from reading or writing files in Documents (or Desktop/Pictures).** Shows up as an "Access is denied" error, or a silent failure to create/update a file, only when the project sits inside one of those folders. Two independent settings can cause this -- check both:
- **Windows Security -> Virus & threat protection -> Manage ransomware protection** -- turn off **Controlled folder access**, or add your terminal/Python/editor to its allowed-apps list.
- **Settings -> Privacy & security -> File system** -- make sure the terminal/Python has access.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, diagnoses and cleans the data (week 3's `data_diagnostics.py`/`preprocessing.py`), preprocesses and trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises.
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the pipeline, and isn't tracked in git (see `.gitignore`) since it's generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Push to GitHub via Terminal

Standard workflow, from the project's root folder, with the venv active:
```bash
git add .
git commit -m "short description of what changed"
git push
```

**If `git push` asks for a password and rejects your normal GitHub password:** GitHub no longer accepts account passwords for git over HTTPS -- you need a **Personal Access Token (PAT)** instead.
1. On GitHub: **Settings -> Developer settings -> Personal access tokens -> Tokens (classic)** -> **Generate new token**, with at least `repo` scope.
2. When `git push` prompts for a password, paste the token instead (username stays your GitHub username).
3. So you're not asked every time: `git config --global credential.helper manager` (Windows, usually already set up by Git for Windows) or `git config --global credential.helper store` (caches it in plaintext -- fine on a personal machine, not a shared one).

Alternative: set up an SSH key once (`ssh-keygen -t ed25519`, then add the public key under **GitHub -> Settings -> SSH and GPG keys**) and use the repo's SSH remote URL (`git@github.com:...`) instead of HTTPS -- no token to manage or renew.

## Dataset

See `data/README.md`.
