# MINE Entropy

Mutual Information Neural Estimation via Entropy Estimation

![MI](./img/MI.png)

## Dependencies

See [binder/requirements.txt](./binder/requirements.txt)

## Show results in Jupyter notebook

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccha23/MINE-Entropy/master?urlpath=lab/tree/results.ipynb)

## Experiment in Jupyter notebook

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccha23/MINE-Entropy/master?urlpath=lab/tree/main.ipynb)

## Rerun experiments

```bash
cd ..
# check the configuration in settings.py
python -m minee.main
```

## Extend the code

### How to add new estimators
- go to ```model/``` 
- create a new class (see mine.py/linear_regression.py for example)
- must contain the predict function (given a N by 2 matrix, return the MI estimator (float))
- go to settings.py, add the corresponding configuration in the variable *model*

### How to add new synthetic data
- go to ```data/``` 
- create a new class (see bimodal.py for example)
- must contain the two property: *data* and *ground_truth* (see bimodal.py for example)
- go to settings.py, add the corresponding configuration in the variable *data*
