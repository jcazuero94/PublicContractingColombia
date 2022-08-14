import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def split_contract_value(secop_clean: pd.DataFrame, features_text: int):
    """Creates train, cv and test dataset for the model that predicts
    contract value based on contract description and other features
    """
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
    # Train cv and test split
    ind_split = np.random.permutation(secop_clean.index)
    train = secop_clean.loc[ind_split].iloc[: int(len(secop_clean) * 0.7)].copy()
    cv = (
        secop_clean.loc[ind_split]
        .iloc[int(len(secop_clean) * 0.7) : int(len(secop_clean) * 0.85)]
        .copy()
    )
    test = secop_clean.loc[ind_split].iloc[int(len(secop_clean) * 0.85) :].copy()
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
