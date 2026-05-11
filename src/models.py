from __future__ import annotations

from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR


RANDOM_STATE = 42


def get_classic_models() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=12.0),
        "lasso": Lasso(alpha=0.0005, max_iter=10000, random_state=RANDOM_STATE),
        "elastic_net": ElasticNet(
            alpha=0.0008,
            l1_ratio=0.7,
            max_iter=10000,
            random_state=RANDOM_STATE,
        ),
        "svr_rbf": SVR(C=20.0, epsilon=0.01, gamma="scale"),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            max_depth=16,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=700,
            learning_rate=0.025,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=0.2,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        ),
    }
