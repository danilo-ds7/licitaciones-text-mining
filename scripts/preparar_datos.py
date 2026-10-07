"""Convierte la descarga JSON (API Mercado Público + embeddings) en los archivos del repositorio."""
import json, sys, re
import numpy as np, pandas as pd

src, out = sys.argv[1], sys.argv[2]
J = json.load(open(src, encoding='utf-8'))

TIPOS = {'L1':'Licitación pública < 100 UTM','LE':'Licitación pública 100-1.000 UTM','LP':'Licitación pública 1.000-2.000 UTM',
         'LQ':'Licitación pública 2.000-5.000 UTM','LR':'Licitación pública > 5.000 UTM','E2':'Licitación privada < 100 UTM',
         'CO':'Licitación privada 100-1.000 UTM','B2':'Licitación privada 1.000-2.000 UTM','H2':'Licitación privada 2.000-5.000 UTM',
         'I2':'Licitación privada > 5.000 UTM','LS':'Licitación pública servicios personales especializados','O1':'Licitación pública de obras',
         'R1':'Orden de compra < 3 UTM'}
ESTADOS = {5:'Publicada',6:'Cerrada',7:'Desierta',8:'Adjudicada',18:'Revocada',19:'Suspendida'}

rows=[]
for x in J['lista']:
    cod=x['CodigoExterno']; m=re.search(r'-([A-Z0-9]{2})(\d{2})$', cod)
    rows.append(dict(codigo=cod, nombre=(x.get('Nombre') or '').strip(), codigo_estado=x.get('CodigoEstado'),
        estado=ESTADOS.get(x.get('CodigoEstado'),'Otro'), fecha_cierre=x.get('FechaCierre'),
        fecha_consulta=pd.to_datetime(x.get('fecha_lista'),format='%d%m%Y').date().isoformat(),
        tipo=(m.group(1) if m else None)))
df=pd.DataFrame(rows).drop_duplicates('codigo')
df['tipo_descripcion']=df['tipo'].map(TIPOS).fillna('Otro')

det=[]
for cod,L in J['detalle'].items():
    if 'Comprador' in L:   # respuesta cruda de la API
        it=(L.get('Items') or {}).get('Listado') or []
        F=L.get('Fechas') or {}
        d=dict(codigo=cod, descripcion=L.get('Descripcion'), organismo=(L.get('Comprador') or {}).get('NombreOrganismo'),
               region=((L.get('Comprador') or {}).get('RegionUnidad') or '').strip(), comuna=(L.get('Comprador') or {}).get('ComunaUnidad'),
               moneda=L.get('Moneda'), monto_estimado=L.get('MontoEstimado'), fecha_publicacion=F.get('FechaPublicacion'),
               fecha_adjudicacion_estimada=F.get('FechaEstimadaAdjudicacion'), duracion_contrato=L.get('TiempoDuracionContrato'),
               unidad_duracion=L.get('UnidadTiempoDuracionContrato'), es_renovable=L.get('EsRenovable'), n_items=len(it),
               categorias=' | '.join(sorted({i.get('Categoria') or '' for i in it})), productos=' | '.join(i.get('NombreProducto') or '' for i in it),
               items_descripcion=' | '.join((i.get('Descripcion') or '').strip() for i in it))
    else:                  # formato resumido (primeros registros)
        d={k:L.get(k) for k in ['descripcion','organismo','region','comuna','moneda','monto_estimado','fecha_publicacion',
            'fecha_adjudicacion_estimada','duracion_contrato','unidad_duracion','es_renovable','n_items','categorias','productos','items_descripcion']}
        d['codigo']=cod
    det.append(d)
dd=pd.DataFrame(det)
df=df.merge(dd,on='codigo',how='left')
df['tiene_detalle']=df['descripcion'].notna()
df.to_csv(f'{out}/licitaciones.csv',index=False)

idx={c:i for i,c in enumerate(J['codigos_vec'])}
V=np.array([J['vec'][idx[c]] for c in df['codigo']],dtype=np.float32)
V/=np.linalg.norm(V,axis=1,keepdims=True)
np.save(f'{out}/embeddings.npy',V.astype(np.float16))
print(df.shape, V.shape, df['tiene_detalle'].sum())
print(df['estado'].value_counts().to_dict()); print(df['tipo'].value_counts().head(8).to_dict())
