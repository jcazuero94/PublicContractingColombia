import pandas as pd
from sodapy import Socrata
from typing import Dict
import datetime
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

CODE_INTEGRATED = "rpmr-utcd"
CODE_SECOPII = "p6dx-8zbt"


def secop_2_log():
    """Creates dictionary of secop2 entities for extraction"""
    client = Socrata("www.datos.gov.co", None)
    request = client.get(CODE_SECOPII, select="distinct nit_entidad")
    nits_list = [x["nit_entidad"] for x in request]
    return {x: {"req": 0, "date": 0} for x in nits_list}


def secop_int_log():
    """Creates dictionary of secop entities for extraction"""
    client = Socrata("www.datos.gov.co", None)
    request = client.get(CODE_INTEGRATED, select="distinct nit_de_la_entidad")
    nits_list = [x["nit_de_la_entidad"] for x in request]
    return {x: {"req": 0, "date": 0} for x in nits_list}


def secop_2_extraction(secop2_log: Dict):
    """Extract secop 2 contracts"""
    # Spark setup
    spark = SparkSession.builder.getOrCreate()
    sql_ctx = SQLContext(spark.sparkContext)
    # Nit to extract. If all nits have been extracted then the oldest extraction is updated
    nits_to_extract = [k for k in secop2_log.keys() if secop2_log[k]["req"] == 0]
    if len(nits_to_extract) > 0:
        nit_to_extract = nits_to_extract[0]
    else:
        list_dates = [(secop2_log[k]["date"], k) for k in secop2_log.keys()]
        list_dates.sort()
        nit_to_extract = list_dates[0][1]
    # Request
    client = Socrata("www.datos.gov.co", None)
    lim = 2000
    offset = lim
    print(f"req - {offset-lim} - {datetime.datetime.now()}")
    request = client.get(
        CODE_SECOPII, limit=lim, where='nit_entidad = "' + nit_to_extract + '"'
    )
    request_df = pd.DataFrame.from_records(request)
    results_df = request_df.copy()
    while len(request_df) > 0:
        print(f"req - {offset} - {datetime.datetime.now()}")
        request = client.get(
            CODE_SECOPII,
            limit=lim,
            offset=offset,
            where='nit_entidad = "' + nit_to_extract + '"',
        )
        request_df = pd.DataFrame.from_records(request)
        results_df = pd.concat([results_df, request_df], ignore_index=True)
        offset += lim
    # Fix nulls
    results_df.fillna("", inplace=True)
    # results_df[[c for c in results_df.columns if "fecha" in c]] = results_df[
    #     [c for c in results_df.columns if "fecha" in c]
    # ].fillna("")
    # Drop useless columns
    results_df.drop("urlproceso", axis=1, inplace=True)
    # Update log
    secop2_log[nit_to_extract]["req"] = 1
    secop2_log[nit_to_extract]["date"] = str(datetime.datetime.now())
    return secop2_log, sql_ctx.createDataFrame(results_df)
