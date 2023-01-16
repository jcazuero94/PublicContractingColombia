import pandas as pd
import numpy as np
from sodapy import Socrata
from typing import Dict
import datetime
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from secop.pipelines.data_engineering.utilities import (
    dic_schemas,
    _get_nits_to_extract,
    _remove_tildes,
    _clean_modalidad_contratacion,
    _clean_modalidad_contratacion_2,
    _clean_tipo_contrato,
    _to_int,
    _clean_tipodocproveecor,
)
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf
import nltk
import string
import re
import simplemma

CODE_INTEGRATED = "rpmr-utcd"
CODE_SECOPII = "p6dx-8zbt"
CODE_SECOPII_CONT = "jbjy-vk9h"


def secop_log(code: str, col_part: str):
    """Creates dictionary of secop entities for extraction"""
    client = Socrata("www.datos.gov.co", None)
    request = client.get(code, select=f"distinct {col_part}", limit=100000)
    nits_list = [x[col_part] for x in request]
    return {x: {"req": 0, "date": 0, "success": 0} for x in nits_list}


def secop_extraction(
    log: Dict, num_nits_to_extract: int, code: str, col_part: str, schema_key: str
):
    """Extract secop contracts"""
    schema = dic_schemas[schema_key]
    spark = SparkSession.builder.getOrCreate()
    sql_ctx = SQLContext(spark.sparkContext)
    # Nit to extract. If all nits have been extracted then the oldest extraction is updated
    nits_to_extract = _get_nits_to_extract(log, num_nits_to_extract)
    # Request
    client = Socrata("www.datos.gov.co", None)
    lim = 4000
    offset = lim
    print(f"req - {offset-lim} - {datetime.datetime.now()}")
    request = client.get(
        code,
        limit=lim,
        select=", ".join(schema.fieldNames()),
        where=col_part + ' in ("' + '","'.join(nits_to_extract) + '")',
    )
    request_df = pd.DataFrame.from_records(request)
    results_df = request_df.copy()
    while len(request_df) > 0:
        print(f"req - {offset} - {datetime.datetime.now()}")
        request = client.get(
            code,
            limit=lim,
            offset=offset,
            select=", ".join(schema.fieldNames()),
            where=col_part + ' in ("' + '","'.join(nits_to_extract) + '")',
        )
        request_df = pd.DataFrame.from_records(request)
        results_df = pd.concat([results_df, request_df], ignore_index=True)
        offset += lim

    # Fix nulls
    results_df.fillna("", inplace=True)
    # Adds columns from schema not received
    for c in set(schema.fieldNames()).difference(results_df.columns):
        results_df[c] = ""
    for n in nits_to_extract:
        log[n]["req"] = 1
        log[n]["date"] = str(datetime.datetime.now())
    success = 1
    for c in set(results_df.columns).intersection(
        [s for s in schema.fieldNames() if str(schema[s].dataType) == "DateType"]
    ):
        results_df[c] = pd.to_datetime(
            results_df[c].replace("", pd.NaT), errors="coerce"
        )
    results_df = results_df[schema.fieldNames()]
    try:
        result_spark = sql_ctx.createDataFrame(results_df, schema=schema)
    except IndexError:
        result_spark = sql_ctx.createDataFrame([], schema)
        success = 0
    for n in nits_to_extract:
        log[n]["success"] = success
    return result_spark, log


def clean_secop_int(secop_int: SparkDataFrame):
    """Clean secop integrated database"""
    # To lower case and remove spainsh accent
    for c in [
        "nivel_entidad",
        "nom_raz_social_contratista",
        "departamento_entidad",
        "municipio_entidad",
        "objeto_a_contratar",
        "objeto_del_proceso",
        "nivel_entidad",
        "estado_del_proceso",
        "modalidad_de_contrataci_n",
        "tipo_de_contrato",
        "nombre_de_la_entidad",
    ]:
        secop_int = secop_int.withColumn(c, F.lower(col(c)))
        if c in [
            "nivel_entidad",
            "nom_raz_social_contratista",
            "departamento_entidad",
            "municipio_entidad",
            "estado_del_proceso",
            "modalidad_de_contrataci_n",
            "tipo_de_contrato",
            "nombre_de_la_entidad",
        ]:
            secop_int = secop_int.withColumn(c, udf(_remove_tildes)(col(c)))
    secop_int = secop_int.withColumn(
        "modalidad_de_contratacion",
        udf(_clean_modalidad_contratacion)(col("modalidad_de_contrataci_n")),
    )
    secop_int = secop_int.drop("modalidad_de_contrataci_n")
    secop_int = secop_int.withColumn(
        "tipo_de_contrato",
        udf(_clean_tipo_contrato)(col("tipo_de_contrato")),
    )
    secop_int = secop_int.withColumn(
        "valor_contrato", col("valor_contrato").cast("integer")
    )
    secop_int = secop_int.withColumn(
        "nit_de_la_entidad",
        udf(lambda x: int(x.replace(".", "").replace(" ", "").split("-")[0][:9]))(
            col("nit_de_la_entidad")
        ),
    )
    return secop_int


def clean_secop_2(secop_2: pd.DataFrame):
    """Clean secop 2 dataset"""
    # To lower case and remove spainsh accent
    COLS_TILDES_LOWER = [
        "entidad",
        "departamento_entidad",
        "ciudad_entidad",
        "ordenentidad",
        "fase",
        "modalidad_de_contratacion",
        "unidad_de_duracion",
        "estado_del_procedimiento",
        "departamento_proveedor",
        "ciudad_proveedor",
        "nombre_del_adjudicador",
        "nombre_del_proveedor",
    ]
    secop_2[COLS_TILDES_LOWER] = (
        secop_2[COLS_TILDES_LOWER].applymap(lambda x: _remove_tildes(x.lower())).values
    )
    secop_2 = secop_2[
        secop_2["modalidad_de_contratacion"]
        != "solicitud de informacion a los proveedores"
    ].copy()
    secop_2 = secop_2[secop_2["estado_del_procedimiento"] == "adjudicado"].copy()
    secop_2["nit_entidad"] = (
        secop_2["nit_entidad"]
        .astype(int)
        .apply(lambda x: int(np.floor(x / 10)) if x >= 1000000000 else x)
    )
    secop_2["modalidad_de_contratacion"] = secop_2["modalidad_de_contratacion"].apply(
        _clean_modalidad_contratacion_2
    )
    for c in [
        "proveedores_invitados",
        "proveedores_con_invitacion",
        "respuestas_al_procedimiento",
        "respuestas_externas",
        "conteo_de_respuestas_a_ofertas",
        "proveedores_unicos_con",
        "duracion",
        "valor_total_adjudicacion",
        "precio_base",
    ]:
        secop_2[c] = secop_2[c].astype(int)
    map_duracion = {"dias": 1, "meses": 30, "años": 365, "nd": 0}
    secop_2["duracion_dias"] = secop_2[["duracion", "unidad_de_duracion"]].apply(
        lambda row: row["duracion"] * map_duracion[row["unidad_de_duracion"]], axis=1
    )
    # TODO Drop columns not used on request
    secop_2.drop(
        [
            "duracion",
            "unidad_de_duracion",
            "adjudicado",
            "tipo_de_contrato",
            "subtipo_de_contrato",
            "visualizaciones_del",
            "proveedores_que_manifestaron",
            "estado_del_procedimiento",
            "fase",
        ],
        axis=1,
        inplace=True,
    )
    secop_2["nit_del_proveedor_adjudicado"] = (
        secop_2["nit_del_proveedor_adjudicado"].apply(_to_int).astype(str)
    )
    secop_2["precio_base"] = secop_2["precio_base"].apply(lambda x: max(x, 0))
    secop_2["valor_total_adjudicacion"] = secop_2.apply(
        lambda row: row["precio_base"]
        if row["valor_total_adjudicacion"] == 0
        else row["valor_total_adjudicacion"],
        axis=1,
    )
    secop_2 = secop_2[secop_2["valor_total_adjudicacion"] != 0].copy()
    secop_2.drop("precio_base", axis=1, inplace=True)
    secop_2.dropna(subset="fecha_de_publicacion_del", inplace=True)
    secop_2 = secop_2[
        ~secop_2["nombre_del_procedimiento"].apply(
            lambda x: ("convenio interadministrativo" == x.lower()[:29])
            or (" copia" == x.lower()[-6:])
        )
    ].copy()
    secop_2["publication_year"] = pd.DatetimeIndex(
        secop_2["fecha_de_publicacion_del"]
    ).year
    secop_2.drop_duplicates(inplace=True)
    return secop_2


def clean_secop_2_cont(secop_2: pd.DataFrame, economia_departamentos: pd.DataFrame):
    """Cleans dataset of secop 2 contracts"""
    # Remove tildes and transform to lower case
    COLS_TILDES_LOWER = [
        "nombre_entidad",
        "departamento",
        "orden",
        "sector",
        "entidad_centralizada",
        "estado_contrato",
        "modalidad_de_contratacion",
        "tipodocproveedor",
        "proveedor_adjudicado",
        "tipo_de_identificaci_n_representante_legal",
    ]
    secop_2[COLS_TILDES_LOWER] = (
        secop_2[COLS_TILDES_LOWER].applymap(lambda x: _remove_tildes(x.lower())).values
    )
    # And only to lower case for not categorical
    secop_2["descripcion_del_proceso"] = secop_2["descripcion_del_proceso"].apply(
        str.lower
    )
    secop_2["objeto_del_contrato"] = secop_2["objeto_del_contrato"].apply(str.lower)
    secop_2["modalidad_de_contratacion"] = secop_2["modalidad_de_contratacion"].apply(
        _clean_modalidad_contratacion_2
    )
    # Dropna
    secop_2.dropna(subset=["fecha_de_firma", "fecha_de_fin_del_contrato"], inplace=True)
    # Process nit
    secop_2["nit_entidad"] = (
        secop_2["nit_entidad"]
        .astype(int)
        .apply(lambda x: int(np.floor(x / 10)) if x >= 1000000000 else x)
    )
    # To int
    for c in ["dias_adicionados", "valor_del_contrato", "documento_proveedor"]:
        secop_2[c] = secop_2[c].apply(_to_int)
    # Remove convenios interadministrativos
    secop_2 = secop_2[
        ~secop_2["descripcion_del_proceso"].apply(
            lambda x: x[:29] == "convenio interadministrativo"
        )
    ].copy()
    secop_2 = secop_2[
        ~secop_2["proveedor_adjudicado"].isin(secop_2["nombre_entidad"].unique())
    ].copy()
    # CLean tipo de proveedor
    secop_2["tipodocproveedor"] = secop_2["tipodocproveedor"].apply(
        _clean_tipodocproveecor
    )
    # Remove rows with problems
    secop_2 = secop_2[secop_2["valor_del_contrato"] > 0].copy()
    secop_2 = secop_2[secop_2["nombre_entidad"] != "viviana bravo rivas"].copy()
    # Log value to decrease skewness
    secop_2["log_valor_del_contrato"] = secop_2["valor_del_contrato"].apply(np.log)
    # Text processing
    stopwords = nltk.corpus.stopwords.words("spanish")
    stopwords = [
        "".join([s for s in w if s not in string.punctuation]) for w in stopwords
    ]

    def text_preprocessing(sentence: str):
        # Replace newline and dash
        sentence = re.sub(r"\n|-", " ", sentence)
        # Delete punctuation
        sentence = "".join([s for s in sentence if s not in string.punctuation])
        # Remove stopwords
        sentence = [w for w in sentence.split(" ") if w not in stopwords and w != ""]
        # Lemmatize
        return " ".join([simplemma.lemmatize(t, lang="es") for t in sentence])

    secop_2["full_contract_description"] = (
        secop_2["descripcion_del_proceso"] + " " + secop_2["objeto_del_contrato"]
    )
    secop_2["full_contract_description"] = secop_2["full_contract_description"].apply(
        text_preprocessing
    )
    # Duration
    secop_2["duration_days"] = secop_2.apply(
        lambda row: max(
            0, (row["fecha_de_fin_del_contrato"] - row["fecha_de_firma"]).days
        )
        if pd.isna(row["fecha_de_inicio_del_contrato"])
        else (
            row["fecha_de_fin_del_contrato"] - row["fecha_de_inicio_del_contrato"]
        ).days,
        axis=1,
    )
    secop_2 = secop_2[secop_2["duration_days"] >= 0].copy()
    # Add economic information
    economia_departamentos["Departamento"] = economia_departamentos[
        "Departamento"
    ].apply(lambda x: _remove_tildes(x.lower()))
    economia_departamentos["Departamento"] = economia_departamentos[
        "Departamento"
    ].replace(
        {
            "san andres, providencia y santa catalina (archipielago)": "san andres, providencia y santa catalina",
            "bogota d.c.": "distrito capital de bogota",
        }
    )
    economia_departamentos.drop("Codigo", axis=1, inplace=True)
    for c in economia_departamentos.columns:
        if c not in ["Departamento", "Poblacion"]:
            economia_departamentos[c] = (
                economia_departamentos[c] / economia_departamentos["Poblacion"]
            )
    secop_2 = pd.merge(
        secop_2,
        economia_departamentos,
        how="left",
        left_on="departamento",
        right_on="Departamento",
    )
    secop_2.drop("Departamento", axis=1, inplace=True)
    for c in economia_departamentos.columns:
        if c != "Departamento":
            secop_2[c] = secop_2[c].fillna(value=secop_2[c].mean())
    secop_2["days_ref"] = secop_2["fecha_de_firma"].apply(
        lambda x: (x - datetime.date(2000, 1, 1)).days
    )
    secop_2 = secop_2[
        (
            (secop_2["documento_proveedor"] > 1000000)
            & (secop_2["documento_proveedor"] < 2000000000)
        )
        | (secop_2["tipodocproveedor"] != "cedula de ciudadania")
    ].copy()
    secop_2["fecha_de_inicio_del_contrato"] = secop_2.apply(
        lambda row: min(
            row["fecha_de_fin_del_contrato"],
            row["fecha_de_firma"]
            if pd.isna(row["fecha_de_inicio_del_contrato"])
            else row["fecha_de_inicio_del_contrato"],
        ),
        axis=1,
    )
    secop_2["urlproceso"] = secop_2["urlproceso"].apply(lambda x: x[5:-1])
    secop_2["documento_proveedor"] = secop_2["documento_proveedor"].astype(str)
    return secop_2.reset_index(drop=True).reset_index()
