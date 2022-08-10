COLS_SEC_2 = [
    "entidad",
    "departamento_entidad",
    "ciudad_entidad",
    "ordenentidad",
    "codigo_pci",
    "id_del_proceso",
    "referencia_del_proceso",
    "ppi",
    "nombre_del_procedimiento",
    "descripci_n_del_procedimiento",
    "fase",
    "precio_base",
    "modalidad_de_contratacion",
    "duracion",
    "unidad_de_duracion",
    "ciudad_de_la_unidad_de",
    "nombre_de_la_unidad_de",
    "proveedores_invitados",
    "proveedores_con_invitacion",
    "visualizaciones_del",
    "proveedores_que_manifestaron",
    "respuestas_al_procedimiento",
    "respuestas_externas",
    "conteo_de_respuestas_a_ofertas",
    "proveedores_unicos_con",
    "estado_del_procedimiento",
    "id_estado_del_procedimiento",
    "adjudicado",
    "id_adjudicacion",
    "codigoproveedor",
    "departamento_proveedor",
    "ciudad_proveedor",
    "valor_total_adjudicacion",
    "nombre_del_adjudicador",
    "nombre_del_proveedor",
    "nit_del_proveedor_adjudicado",
    "codigo_principal_de_categoria",
    "estado_de_apertura_del_proceso",
    "tipo_de_contrato",
    "subtipo_de_contrato",
    "categorias_adicionales",
    "codigo_entidad",
    "estadoresumen",
    "fecha_de_publicacion_del",
    "fecha_de_ultima_publicaci",
    "fecha_de_publicacion_fase_3",
    "fecha_de_recepcion_de",
    "fecha_de_apertura_efectiva",
    "nit_entidad",
]
COLS_INT = [
    "nivel_entidad",
    "nombre_de_la_entidad",
    "nit_de_la_entidad",
    "estado_del_proceso",
    "modalidad_de_contrataci_n",
    "objeto_a_contratar",
    "tipo_de_contrato",
    "numero_del_contrato",
    "numero_de_proceso",
    "valor_contrato",
    "nom_raz_social_contratista",
    "departamento_entidad",
    "municipio_entidad",
    "objeto_del_proceso",
    "tipo_contrato",
    "origen",
    "fecha_de_firma_del_contrato",
    "fecha_inicio_ejecucion",
    "fecha_fin_ejecucion",
]


def _get_nit_to_extract(log):
    nits_to_extract = [k for k in log.keys() if log[k]["req"] == 0]
    if len(nits_to_extract) > 0:
        nit_to_extract = nits_to_extract[0]
    else:
        list_failed = [
            (log[k]["date"], k) for k in log.keys() if log[k]["success"] == 0
        ]
        list_dates = (
            [(log[k]["date"], k) for k in log.keys()]
            if len(list_failed) == 0
            else list_failed
        )
        list_dates.sort()
        nit_to_extract = list_dates[0][1]
    return nit_to_extract
