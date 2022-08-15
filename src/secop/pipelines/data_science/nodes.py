import warnings
from xmlrpc.client import Boolean

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from secop.pipelines.data_science.utilities import _vectorize, _padSer
import tensorflow as tf
from tensorflow import keras
from gensim import downloader
from sklearn.metrics import mean_squared_error
import datetime
import os


def split_contract_value(secop_clean: pd.DataFrame, features_text: int):
    """Creates train, cv and test dataset for the model that predicts
    contract value based on contract description and other features
    """
    # Select columns
    secop_clean = secop_clean[
        [
            "index",
            "full_contract_description",
            "proceso_de_compra",
            "orden",
            "modalidad_de_contratacion",
            "duration_days",
            "Agricultura",
            "Hidrocarburos",
            "Manufactura",
            "ServiciosPublicos",
            "Construccion",
            "Comercio",
            "Comunicaciones",
            "Financiero",
            "Inmobiliaria",
            "Profesionales",
            "AdministracionPublica",
            "Artisticas",
            "ValorAgregado",
            "Impuestos",
            "PIB",
            "Poblacion",
            "log_valor_del_contrato",
            "dias_adicionados",
            "days_ref",
        ]
    ].copy()
    # Get dummies for categorical features
    secop_clean = pd.get_dummies(
        secop_clean, columns=["orden", "modalidad_de_contratacion"], drop_first=True
    )
    # Train cv and test split
    ids_processes = np.random.permutation(secop_clean["proceso_de_compra"].unique())
    train = secop_clean[
        secop_clean["proceso_de_compra"].isin(
            ids_processes[: int(len(ids_processes) * 0.7)]
        )
    ].copy()
    cv = secop_clean[
        secop_clean["proceso_de_compra"].isin(
            ids_processes[
                int(len(ids_processes) * 0.7) : int(len(ids_processes) * 0.85)
            ]
        )
    ].copy()
    test = secop_clean[
        secop_clean["proceso_de_compra"].isin(
            ids_processes[int(len(ids_processes) * 0.85) :]
        )
    ].copy()
    train.drop("proceso_de_compra", axis=1, inplace=True)
    cv.drop("proceso_de_compra", axis=1, inplace=True)
    test.drop("proceso_de_compra", axis=1, inplace=True)
    del secop_clean
    # TfIdf vectorization
    features_tfid = [f"feat_text_{i+1}" for i in range(features_text)]
    vectorizer = TfidfVectorizer(max_features=features_text)
    text_df_train = pd.DataFrame(
        columns=features_tfid,
        data=np.array(
            vectorizer.fit_transform(train["full_contract_description"]).todense()
        ),
    )
    train = pd.merge(
        train, text_df_train, how="inner", left_index=True, right_index=True
    )
    del text_df_train
    text_df_cv = pd.DataFrame(
        columns=features_tfid,
        data=np.array(vectorizer.transform(cv["full_contract_description"]).todense()),
    )
    text_df_test = pd.DataFrame(
        columns=features_tfid,
        data=np.array(
            vectorizer.transform(test["full_contract_description"]).todense()
        ),
    )
    cv = pd.merge(cv, text_df_cv, how="inner", left_index=True, right_index=True)
    del text_df_cv
    test = pd.merge(test, text_df_test, how="inner", left_index=True, right_index=True)
    del text_df_test
    return (
        train.drop(["full_contract_description"], axis=1),
        cv.drop(["full_contract_description"], axis=1),
        test.drop(["full_contract_description"], axis=1),
        train.drop(features_tfid, axis=1),
        cv.drop(features_tfid, axis=1),
        test.drop(features_tfid, axis=1),
    )


def prepare_clusters_contract(secop_clean: pd.DataFrame, features_text: int):
    """Prepares dataset for contract cluster construction"""
    # Select columns
    secop_clean = secop_clean[
        [
            "index",
            "full_contract_description",
            "orden",
            "modalidad_de_contratacion",
            "duration_days",
            "Agricultura",
            "Hidrocarburos",
            "Manufactura",
            "ServiciosPublicos",
            "Construccion",
            "Comercio",
            "Comunicaciones",
            "Financiero",
            "Inmobiliaria",
            "Profesionales",
            "AdministracionPublica",
            "Artisticas",
            "ValorAgregado",
            "Impuestos",
            "PIB",
            "Poblacion",
            "log_valor_del_contrato",
            "dias_adicionados",
            "days_ref",
        ]
    ].copy()
    # Get dummies for categorical features
    secop_clean = pd.get_dummies(
        secop_clean, columns=["orden", "modalidad_de_contratacion"], drop_first=True
    )
    cols_not_to_scale = ["log_valor_del_contrato", "full_contract_description", "index"]
    standard_scaler = StandardScaler()
    secop_clean[
        [x for x in secop_clean.columns if x not in cols_not_to_scale]
    ] = standard_scaler.fit_transform(secop_clean.drop(cols_not_to_scale, axis=1))
    vectorizer = TfidfVectorizer(max_features=features_text)
    text_df = pd.DataFrame(
        columns=[f"feat_text_{i+1}" for i in range(features_text)],
        data=np.array(
            vectorizer.fit_transform(secop_clean["full_contract_description"]).todense()
        ),
    )
    secop_clean = pd.merge(
        secop_clean, text_df, how="inner", left_index=True, right_index=True
    )
    secop_clean.drop(["full_contract_description"], axis=1, inplace=True)
    return secop_clean


def train_rnn(train: pd.DataFrame, cv: pd.DataFrame, from_checkpoint: bool):
    """Train LSTM"""
    cwd = os.getcwd()
    word_2_vec = downloader.load("word2vec-google-news-300")
    w2v_len = word_2_vec["hola"].shape[0]
    if from_checkpoint:
        model = keras.models.load_model(cwd + "/data/06_models/rnn")
    else:
        inputs = keras.layers.Input(
            shape=(None, len(train.columns) + w2v_len - 3), dtype=tf.float64
        )
        x = keras.layers.Masking(
            mask_value=0.0,
            input_shape=(None, len(train.columns) + w2v_len - 3),
        )(inputs)
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5)
        )(x)
        x = keras.layers.Dense(10, activation="relu")(x)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.RMSprop(),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.RootMeanSquaredError()],
        )
        print(model.summary())
    num_epochs = 5
    batch_size = 3000
    num_batches = np.ceil(len(train) / batch_size)
    num_batches_cv = np.ceil(len(cv) / batch_size)
    w2v_len = word_2_vec["hola"].shape[0]
    rmses = []
    rmses_cv = []
    min_rmse = np.inf
    cont = 0
    for ep in range(num_epochs):
        print(f"Ep:{ep} - {datetime.datetime.today().strftime('%H:%M:%S')}")
        for i in range(int(num_batches)):
            train_other = (
                train.drop(
                    ["index", "full_contract_description", "log_valor_del_contrato"],
                    axis=1,
                )
                .iloc[i * batch_size : (i + 1) * batch_size]
                .values
            )
            train_text = (
                train["full_contract_description"]
                .iloc[i * batch_size : (i + 1) * batch_size]
                .apply(lambda x: _vectorize(x, word_2_vec))
            )
            train_text = np.stack(_padSer(train_text, w2v_len))
            x_train = np.concatenate(
                [
                    train_text,
                    np.tile(
                        train_other.reshape(
                            (train_other.shape[0], 1, train_other.shape[1])
                        ),
                        (1, train_text.shape[1], 1),
                    ),
                ],
                axis=2,
            )
            y_train = (
                train["log_valor_del_contrato"]
                .iloc[i * batch_size : (i + 1) * batch_size]
                .values
            )
            rmses += [
                model.fit(x_train, y_train, verbose=0).history[
                    "root_mean_squared_error"
                ][0]
            ]

            if i % 20 == 0:
                print(
                    f"Ep:{ep} - it:{i} - rmse:{np.mean(rmses[-20:])} - {datetime.datetime.today().strftime('%H:%M:%S')}"
                )

        cv_pred = np.array([])
        for i in range(int(num_batches_cv)):
            cv_other = (
                cv.drop(
                    ["index", "full_contract_description", "log_valor_del_contrato"],
                    axis=1,
                )
                .iloc[i * batch_size : (i + 1) * batch_size]
                .values
            )
            cv_text = (
                cv["full_contract_description"]
                .iloc[i * batch_size : (i + 1) * batch_size]
                .apply(lambda x: _vectorize(x, word_2_vec))
            )
            cv_text = np.stack(_padSer(cv_text, w2v_len))
            x_cv = np.concatenate(
                [
                    cv_text,
                    np.tile(
                        cv_other.reshape((cv_other.shape[0], 1, cv_other.shape[1])),
                        (1, cv_text.shape[1], 1),
                    ),
                ],
                axis=2,
            )
            if len(cv_pred) > 0:
                cv_pred = np.concatenate([cv_pred, model(x_cv)])
            else:
                cv_pred = model(x_cv)

        y_cv = cv["log_valor_del_contrato"].values
        rmses_cv += [mean_squared_error(y_cv, cv_pred)]
        print(f"Ep:{ep} - cv rmse:{rmses_cv[-1]}")
        if rmses_cv[-1] < min_rmse:
            min_rmse = rmses_cv[-1]
            cont = 0
            model.save(cwd + "/data/06_models/rnn")
        else:
            cont += 1
        if cont > 5:
            break
