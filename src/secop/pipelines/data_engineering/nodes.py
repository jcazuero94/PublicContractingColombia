import pandas as pd
from sodapy import Socrata
from typing import Dict
import datetime
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from secop.pipelines.data_engineering.utilities import (
    COLS_SEC_2,
    COLS_INT,
    _get_nit_to_extract,
    _remove_tildes,
    _clean_modalidad_contratacion,
    _clean_tipo_contrato,
)
from pyspark.sql.types import StructType
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf

CODE_INTEGRATED = "rpmr-utcd"
CODE_SECOPII = "p6dx-8zbt"


def secop_2_log():
    """Creates dictionary of secop2 entities for extraction"""
    client = Socrata("www.datos.gov.co", None)
    request = client.get(CODE_SECOPII, select="distinct nit_entidad")
    nits_list = [x["nit_entidad"] for x in request]
    return {x: {"req": 0, "date": 0, "success": 0} for x in nits_list}


def secop_int_log():
    """Creates dictionary of secop entities for extraction"""
    client = Socrata("www.datos.gov.co", None)
    request = client.get(CODE_INTEGRATED, select="distinct nit_de_la_entidad")
    nits_list = [x["nit_de_la_entidad"] for x in request]
    return {x: {"req": 0, "date": 0, "success": 0} for x in nits_list}


def secop_2_extraction(secop_2_log: Dict):
    """Extract secop 2 contracts"""
    # Spark setup
    spark = SparkSession.builder.getOrCreate()
    sql_ctx = SQLContext(spark.sparkContext)
    # Nit to extract. If all nits have been extracted then the oldest extraction is updated
    nit_to_extract = _get_nit_to_extract(secop_2_log)
    # Request
    client = Socrata("www.datos.gov.co", None)
    lim = 2000
    offset = lim
    print(f"req - {offset-lim} - {datetime.datetime.now()}")
    request = client.get(
        CODE_SECOPII,
        limit=lim,
        select=", ".join(COLS_SEC_2),
        where='nit_entidad = "' + nit_to_extract + '"',
    )
    request_df = pd.DataFrame.from_records(request)
    results_df = request_df.copy()
    while len(request_df) > 0:
        print(f"req - {offset} - {datetime.datetime.now()}")
        request = client.get(
            CODE_SECOPII,
            limit=lim,
            offset=offset,
            select=", ".join(COLS_SEC_2),
            where='nit_entidad = "' + nit_to_extract + '"',
        )
        request_df = pd.DataFrame.from_records(request)
        results_df = pd.concat([results_df, request_df], ignore_index=True)
        offset += lim
    # Fix nulls
    results_df.fillna("", inplace=True)
    try:
        result_spark = sql_ctx.createDataFrame(results_df)
        secop_2_log[nit_to_extract]["success"] = 1
    except IndexError:
        schema = StructType([])
        result_spark = sql_ctx.createDataFrame([], schema)
        secop_2_log[nit_to_extract]["success"] = 0
    secop_2_log[nit_to_extract]["req"] = 1
    secop_2_log[nit_to_extract]["date"] = str(datetime.datetime.now())

    return result_spark, secop_2_log


def secop_int_extraction(secop_int_log: Dict):
    """Extract secop contracts from database secop integrado"""
    # Spark setup
    spark = SparkSession.builder.getOrCreate()
    sql_ctx = SQLContext(spark.sparkContext)
    # Nit to extract. If all nits have been extracted then the oldest extraction is updated
    nit_to_extract = _get_nit_to_extract(secop_int_log)
    # Request
    client = Socrata("www.datos.gov.co", None)
    lim = 2000
    offset = lim
    print(f"req - {offset-lim} - {datetime.datetime.now()}")
    request = client.get(
        CODE_INTEGRATED,
        limit=lim,
        select=", ".join(COLS_INT),
        where='nit_de_la_entidad = "' + nit_to_extract + '"',
    )
    request_df = pd.DataFrame.from_records(request)
    results_df = request_df.copy()
    while len(request_df) > 0:
        print(f"req - {offset} - {datetime.datetime.now()}")
        request = client.get(
            CODE_INTEGRATED,
            limit=lim,
            offset=offset,
            select=", ".join(COLS_INT),
            where='nit_de_la_entidad = "' + nit_to_extract + '"',
        )
        request_df = pd.DataFrame.from_records(request)
        results_df = pd.concat([results_df, request_df], ignore_index=True)
        offset += lim
    # Fix nulls
    results_df.fillna("", inplace=True)
    try:
        result_spark = sql_ctx.createDataFrame(results_df)
        secop_int_log[nit_to_extract]["success"] = 1
    except IndexError:
        schema = StructType([])
        result_spark = sql_ctx.createDataFrame([], schema)
        secop_int_log[nit_to_extract]["success"] = 0
    secop_int_log[nit_to_extract]["req"] = 1
    secop_int_log[nit_to_extract]["date"] = str(datetime.datetime.now())
    return result_spark, secop_int_log


def clean_secop_int(secop_int: SparkDataFrame):
    """Clean secop integrated database"""
    # Drop unused columns
    # TODO include this restriction in the query
    secop_int = secop_int.drop(
        "origen", "tipo_contrato", "numero_del_contrato", "numero_de_proceso"
    )
    # To lower case and remove spainsh accent
    for c in [
        "nom_raz_social_contratista",
        "departamento_entidad",
        "municipio_entidad",
        "objeto_a_contratar",
        "objeto_del_proceso",
        "nivel_entidad",
        "estado_del_proceso",
        "modalidad_de_contrataci_n",
        "tipo_de_contrato",
    ]:
        secop_int = secop_int.withColumn(c, F.lower(col(c)))
        if c in [
            "nom_raz_social_contratista",
            "departamento_entidad",
            "municipio_entidad",
            "estado_del_proceso",
            "modalidad_de_contrataci_n",
            "tipo_de_contrato",
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
        udf(lambda x: int(x.replace(".", "").split("-")[0]))(col("nit_de_la_entidad")),
    )
    return secop_int
