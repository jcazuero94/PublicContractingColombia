import pandas as pd
from sodapy import Socrata
from typing import Dict
import datetime
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from secop.pipelines.data_engineering.utilities import (
    COLS_SEC_2,
    COLS_INT,
    _get_nit_to_extract,
)
from pyspark.sql.types import StructType

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
