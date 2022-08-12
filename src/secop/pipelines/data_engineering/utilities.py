from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DateType,
)


def _get_nits_to_extract(log: dict, num_nits_to_extract: int):
    nits_to_extract = [k for k in log.keys() if log[k]["req"] == 0]
    if len(nits_to_extract) > 0:
        return nits_to_extract[0:num_nits_to_extract]
    else:
        print("All nits previously extracted. Retry failed and update")
        list_failed = [
            (log[k]["date"], k) for k in log.keys() if log[k]["success"] == 0
        ]
        list_dates = (
            [(log[k]["date"], k) for k in log.keys()]
            if len(list_failed) == 0
            else list_failed
        )
        list_dates.sort()
        return [x[1] for x in list_dates][:num_nits_to_extract]


def _remove_tildes(string):
    """Remove spanish accentuation mark for string standarization"""
    return (
        str(string)
        .replace("á", "a")
        .replace("é", "e")
        .replace("ó", "o")
        .replace("í", "i")
        .replace("ú", "u")
    )


def _clean_tipo_contrato(tip: str):
    """Clean and group tipo de contrato"""
    if ("suministro" in tip) or (tip in ["compraventa", "venta muebles"]):
        return "suministro"
    elif ("arrendamiento" in tip) or ("comodato" in tip):
        return "arrendamiento"
    elif tip in [
        "servicios financieros",
        "credito",
        "fiducia",
        "seguros",
        "emprestito",
    ]:
        return "servicios financieros"
    elif tip in [
        "obra",
        "consultoria",
        "prestacion de servicios",
        "interventoria",
        "concesion",
    ]:
        return tip
    else:
        return "Otro"


def _clean_modalidad_contratacion(mod: str):
    """Clean and group modalidad de contratacion"""
    if ("concurso de meritos" in mod) or ("concurso_meritos" in mod):
        return "concurso de meritos abiertos"
    elif "regimen especial" in mod:
        return "regimen especial"
    elif ("minima cuantia" in mod) or ("menor cuantia" in mod):
        return "minima cuantia"
    elif "contratacion directa" in mod:
        return "contratacion directa"
    elif "subasta" in mod:
        return "subasta"
    elif ("licitacion publica" in mod) or ("licitacion obra publica" in mod):
        return "licitacion publica"
    else:
        return "Otro"


schema_secop_2 = StructType(
    [
        StructField("nit_entidad", StringType(), True),
        StructField("entidad", StringType(), True),
        StructField("departamento_entidad", StringType(), True),
        StructField("ciudad_entidad", StringType(), True),
        StructField("ordenentidad", StringType(), True),
        StructField("id_del_proceso", StringType(), True),
        StructField("referencia_del_proceso", StringType(), True),
        StructField("nombre_del_procedimiento", StringType(), True),
        StructField("descripci_n_del_procedimiento", StringType(), True),
        StructField("fase", StringType(), True),
        StructField("precio_base", StringType(), True),
        StructField("modalidad_de_contratacion", StringType(), True),
        StructField("duracion", StringType(), True),
        StructField("unidad_de_duracion", StringType(), True),
        StructField("proveedores_invitados", StringType(), True),
        StructField("proveedores_con_invitacion", StringType(), True),
        StructField("visualizaciones_del", StringType(), True),
        StructField("proveedores_que_manifestaron", StringType(), True),
        StructField("respuestas_al_procedimiento", StringType(), True),
        StructField("respuestas_externas", StringType(), True),
        StructField("conteo_de_respuestas_a_ofertas", StringType(), True),
        StructField("proveedores_unicos_con", StringType(), True),
        StructField("estado_del_procedimiento", StringType(), True),
        StructField("adjudicado", StringType(), True),
        StructField("departamento_proveedor", StringType(), True),
        StructField("ciudad_proveedor", StringType(), True),
        StructField("valor_total_adjudicacion", StringType(), True),
        StructField("nombre_del_adjudicador", StringType(), True),
        StructField("nombre_del_proveedor", StringType(), True),
        StructField("nit_del_proveedor_adjudicado", StringType(), True),
        StructField("tipo_de_contrato", StringType(), True),
        StructField("subtipo_de_contrato", StringType(), True),
        StructField("fecha_de_publicacion_del", DateType(), True),
        StructField("fecha_de_ultima_publicaci", DateType(), True),
        StructField("fecha_de_publicacion_fase_3", DateType(), True),
        StructField("fecha_de_recepcion_de", DateType(), True),
        StructField("fecha_de_apertura_efectiva", DateType(), True),
    ]
)

schema_secop_int = StructType(
    [
        StructField("nivel_entidad", StringType(), True),
        StructField("nombre_de_la_entidad", StringType(), True),
        StructField("nit_de_la_entidad", StringType(), True),
        StructField("estado_del_proceso", StringType(), True),
        StructField("modalidad_de_contrataci_n", StringType(), True),
        StructField("objeto_a_contratar", StringType(), True),
        StructField("tipo_de_contrato", StringType(), True),
        StructField("valor_contrato", StringType(), True),
        StructField("nom_raz_social_contratista", StringType(), True),
        StructField("departamento_entidad", StringType(), True),
        StructField("municipio_entidad", StringType(), True),
        StructField("objeto_del_proceso", StringType(), True),
        StructField("fecha_de_firma_del_contrato", DateType(), True),
        StructField("fecha_inicio_ejecucion", DateType(), True),
        StructField("fecha_fin_ejecucion", DateType(), True),
    ]
)
