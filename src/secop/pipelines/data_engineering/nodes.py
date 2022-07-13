import pandas as pd
from sodapy import Socrata

CODE_INTEGRATED = "rpmr-utcd"
CODE_SECOPII = "p6dx-8zbt"


def secop_2_log():
    """Creates dictionary of secop2 entities for extraction"""
    client = Socrata("www.datos.gov.co", None)
    request = client.get(CODE_SECOPII, select="distinct nit_entidad")
    nits_list = [x["nit_entidad"] for x in request]
    return {x: {"req": 0, "date": 0} for x in nits_list}
