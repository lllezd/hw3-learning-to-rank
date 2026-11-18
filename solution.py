#!/usr/bin/env python
"""Обучение ансамбля и формирование CSV для ДЗ3."""

import argparse
from io import BytesIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from timeit import default_timer as timer
from zipfile import ZIP_DEFLATED, ZipFile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool


SEED = 0
THREADS = min(8, os.cpu_count() or 1)
CATBOOST_PARAMS = dict(
    iterations=600, loss_function='YetiRank', depth=6, learning_rate=0.05,
    l2_leaf_reg=5, random_seed=SEED, thread_count=THREADS,
    eval_metric='NDCG:top=10;type=Exp;denominator=LogPosition',
    allow_writing_files=False, verbose=False,
)
LIGHTGBM_ROUNDS = 650
LISTWISE_EPOCHS = 25
LIGHTGBM_PARAMS = dict(
    objective='lambdarank', label_gain=[0, 1, 3, 7], lambdarank_truncation_level=13,
    learning_rate=0.04, num_leaves=15, max_depth=5, min_data_in_leaf=180,
    lambda_l2=8, max_bin=127, feature_fraction=0.85, deterministic=True,
    force_col_wise=True, num_threads=THREADS, seed=SEED, verbosity=-1,
)


QUERY_ZSCORE_COLUMNS = (
    'bm25f', 'bm25_title', 'bm25_body', 'bm25_url', 'lm_title', 'lm_body',
    'bm25_stem_title', 'bm25_stem_body', 'coverage_stem_title', 'coverage_stem_body',
    'title_trigram_jaccard', 'query_translit_in_url', 'body_matched_first50', 'domain_prior',
)
QUERY_CONTEXT_COLUMNS = (
    'bm25f', 'bm25_title', 'bm25_body', 'coverage_stem_body',
    'coverage_stem_title', 'title_trigram_jaccard', 'idf_coverage_body', 'domain_prior',
)


def add_query_zscores(data):
    """Добавить отклонения от среднего в единицах разброса внутри запроса."""
    columns = list(QUERY_ZSCORE_COLUMNS)
    group = data.groupby('QueryId', sort=False)[columns]
    centered = data.loc[:, columns] - group.transform('mean')
    deviation = group.transform('std', ddof=0)
    zscores = centered.div(deviation.where(deviation > 0)).fillna(0)
    return data.join(zscores.add_suffix('_query_zscore'))


def add_query_context(data):
    """Добавить среднее и максимум признака среди документов запроса."""
    group = data.groupby('QueryId', sort=False)[list(QUERY_CONTEXT_COLUMNS)]
    means = group.transform('mean').add_suffix('_query_mean')
    maxima = group.transform('max').add_suffix('_query_max')
    return data.join([means, maxima])


def prepare_features(data):
    return add_query_context(add_query_zscores(data))


def feature_matrix(data, feature_names=None):
    """Идентификаторы и правильные ответы не становятся признаками модели."""
    if feature_names is None:
        feature_names = [
            column for column in data.columns
            if column not in ('ObjectId', 'QueryId', 'Label')
        ]
    return data.loc[:, feature_names].astype(np.float32)


def load_train(data_dir):
    data = pd.read_csv(Path(data_dir) / 'train.csv')
    # При обучении документы одного запроса идут подряд.
    return data.sort_values('QueryId', kind='stable')


def fit_catboost_ranker(data):
    pool = Pool(feature_matrix(data), label=data.Label, group_id=data.QueryId)
    return CatBoostRanker(**CATBOOST_PARAMS).fit(pool)


def fit_lightgbm_ranker(data):
    import lightgbm as lgb

    group_sizes = data.groupby('QueryId', sort=False).size().to_numpy()
    dataset = lgb.Dataset(feature_matrix(data), label=data.Label, group=group_sizes)
    return lgb.train(LIGHTGBM_PARAMS.copy(), dataset, num_boost_round=LIGHTGBM_ROUNDS)


def listwise_forward(inputs, parameters):
    w1, b1, w2 = parameters
    hidden = np.maximum(inputs @ w1 + b1, 0)
    return hidden, hidden @ w2


def fit_listwise(data):
    """Обучить компактную сеть на распределениях релевантности внутри запросов."""
    raw = feature_matrix(data).to_numpy()
    mean = raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = raw.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale == 0] = 1
    features = np.arcsinh((raw - mean) / scale)

    _, starts, sizes = np.unique(data.QueryId, return_index=True, return_counts=True)
    gains = np.exp2(data.Label.to_numpy()).astype(np.float32) - 1
    totals = np.repeat(np.add.reduceat(gains, starts), sizes)
    targets = np.where(
        totals > 0, gains / np.maximum(totals, 1), 1 / np.repeat(sizes, sizes),
    ).astype(np.float32)

    rng = np.random.default_rng(SEED)
    width = 64
    parameters = [
        (rng.standard_normal((features.shape[1], width)) * np.sqrt(2 / features.shape[1])).astype(np.float32),
        np.zeros(width, dtype=np.float32),
        (rng.standard_normal(width) * 0.03).astype(np.float32),
    ]
    first_moments = [np.zeros_like(parameter) for parameter in parameters]
    second_moments = [np.zeros_like(parameter) for parameter in parameters]
    averages = None
    step = 0

    for _ in range(LISTWISE_EPOCHS):
        permutation = rng.permutation(len(sizes))
        # Порция обучения состоит из целых запросов.
        for offset in range(0, len(sizes), 96):
            groups = permutation[offset:offset + 96]
            batch_sizes = sizes[groups]
            batch_starts = np.r_[0, np.cumsum(batch_sizes)[:-1]]
            rows = np.arange(batch_sizes.sum()) + np.repeat(starts[groups] - batch_starts, batch_sizes)
            inputs = features[rows]
            hidden, scores = listwise_forward(inputs, parameters)
            shifted = scores - np.repeat(np.maximum.reduceat(scores, batch_starts), batch_sizes)
            weights = np.exp(shifted)
            probabilities = weights / np.repeat(np.add.reduceat(weights, batch_starts), batch_sizes)
            gradient = (probabilities - targets[rows]) / len(groups)
            hidden_gradient = gradient[:, None] * parameters[2] * (hidden > 0)
            gradients = (
                inputs.T @ hidden_gradient + 0.0001 * parameters[0],
                hidden_gradient.sum(axis=0),
                hidden.T @ gradient,
            )
            step += 1
            for parameter, first, second, grad in zip(parameters, first_moments, second_moments, gradients):
                first *= 0.9
                first += 0.1 * grad
                second *= 0.999
                second += 0.001 * grad**2
                numerator = 0.001 * (first / (1 - 0.9**step))
                denominator = np.sqrt(second / (1 - 0.999**step)) + 1e-8
                parameter -= numerator / denominator
            if step >= 500:
                if averages is None:
                    averages = [parameter.copy() for parameter in parameters]
                else:
                    for average, parameter in zip(averages, parameters):
                        average += 0.005 * (parameter - average)

    w1, b1, w2 = averages if averages is not None else parameters
    return dict(mean=mean, scale=scale, w1=w1, b1=b1, w2=w2)


def predict_listwise(features, state):
    inputs = np.arcsinh((features.to_numpy(dtype=np.float64) - state['mean']) / state['scale'])
    parameters = tuple(state[name].astype(np.float64) for name in ('w1', 'b1', 'w2'))
    return listwise_forward(inputs, parameters)[1]


def fit_ensemble(data):
    data = prepare_features(data)
    return fit_catboost_ranker(data), fit_lightgbm_ranker(data), fit_listwise(data)


def save_ensemble(path, catboost_model, lightgbm_model, listwise_model):
    with TemporaryDirectory(dir=path.parent) as directory:
        directory = Path(directory)
        catboost_file = directory / 'catboost.cbm'
        catboost_model.save_model(str(catboost_file))
        archive_path = directory / 'model.zip'
        with ZipFile(archive_path, 'w', compression=ZIP_DEFLATED) as archive:
            archive.write(catboost_file, 'catboost.cbm')
            archive.writestr('lightgbm.txt', lightgbm_model.model_to_string())
            buffer = BytesIO()
            np.savez(buffer, **listwise_model)
            archive.writestr('listwise.npz', buffer.getvalue())
        archive_path.replace(path)


def normalize_scores(scores, query_ids):
    values = pd.Series(scores, index=query_ids.index)
    group = values.groupby(query_ids, sort=False)
    deviation = group.transform('std', ddof=0)
    return (values - group.transform('mean')).div(deviation.where(deviation > 0)).fillna(0).to_numpy()


def load_ensemble(path):
    import lightgbm as lgb

    with ZipFile(path) as archive:
        catboost_model = CatBoostRanker().load_model(blob=archive.read('catboost.cbm'))
        lightgbm_model = lgb.Booster(model_str=archive.read('lightgbm.txt').decode('utf-8'))
        with np.load(BytesIO(archive.read('listwise.npz')), allow_pickle=False) as arrays:
            listwise_model = dict(arrays)
    return catboost_model, lightgbm_model, listwise_model


def score_ensemble(data, catboost_model, lightgbm_model, listwise_model):
    data = prepare_features(data)
    features = feature_matrix(data, catboost_model.feature_names_)
    catboost_scores = catboost_model.predict(features, thread_count=THREADS)
    lightgbm_scores = lightgbm_model.predict(
        feature_matrix(data, lightgbm_model.feature_name()), num_threads=THREADS, validate_features=True,
    )
    listwise_scores = predict_listwise(features, listwise_model)
    trees = 0.75 * normalize_scores(catboost_scores, data.QueryId) + 0.25 * normalize_scores(lightgbm_scores, data.QueryId)
    return 0.80 * trees + 0.20 * normalize_scores(listwise_scores, data.QueryId)


def train(data_dir, model_file):
    models = fit_ensemble(load_train(data_dir))
    path = Path(model_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ensemble(path, *models)


def predict(data_dir, model_file, submission_file):
    test = pd.read_csv(Path(data_dir) / 'test.csv')
    scores = score_ensemble(test, *load_ensemble(model_file))
    if not np.isfinite(scores).all():
        raise ValueError('Модель выдала нечисловые или бесконечные оценки')
    if not test['ObjectId'].is_unique:
        raise ValueError('В test.csv повторяются ObjectId')
    submission = pd.DataFrame({'ObjectId': test['ObjectId'], 'Score': scores})
    path = Path(submission_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Learning to rank homework solution')
    parser.add_argument('--train', action='store_true', help='обучить модель и сохранить её в model_file')
    parser.add_argument('--model_file', required=True, help='файл модели')
    parser.add_argument('--submission_file', help='куда записать сабмишн для Kaggle')
    parser.add_argument('data_dir', help='папка с данными соревнования')
    args = parser.parse_args()

    start = timer()

    if args.train:
        train(args.data_dir, args.model_file)
        output = args.model_file
    else:
        if args.submission_file is None:
            parser.error('без --train нужен --submission_file')
        predict(args.data_dir, args.model_file, args.submission_file)
        output = args.submission_file

    print(f'{output} ({timer() - start:.3f} s)')


if __name__ == '__main__':
    main()
