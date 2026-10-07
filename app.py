"""Text mining + Transformers sobre licitaciones reales de Mercado Público.
Módulo V · IA Generativa para la Gestión Industrial · Docente: Danilo Gómez Correa
"""
import collections
import itertools
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

from texto_utils import STOPWORDS_ES, limpiar, quitar_tildes

st.set_page_config(page_title='Text mining de licitaciones', page_icon='📑', layout='wide')

AZUL, NARANJO, VERDE, CELESTE = '#091D44', '#D68014', '#228054', '#3C78BE'
TRAMOS = {'L1': '< 100 UTM', 'LE': '100-1.000 UTM', 'LP': '1.000-2.000 UTM', 'LQ': '2.000-5.000 UTM', 'LR': '> 5.000 UTM'}


# ------------------------------------------------------------------ datos
@st.cache_data
def cargar():
    df = pd.read_csv('data/licitaciones.csv')
    mapa = pd.read_csv('data/mapa_embeddings.csv')
    df = df.merge(mapa, on='codigo', how='left')
    df['tramo'] = df['tipo'].map(TRAMOS).fillna('Otros (privadas, obras, etc.)')
    df['nombre_limpio'] = df['nombre'].fillna('').map(limpiar)
    df['nombre_suave'] = df['nombre'].fillna('').map(lambda t: limpiar(t, quitar_dominio=False))
    E = np.load('data/embeddings.npy').astype(np.float32)
    temas = pd.read_csv('data/topicos_bertopic.csv')
    return df, E, temas


df_all, E_all, temas_bt = cargar()

# ------------------------------------------------------------------ barra lateral
st.sidebar.title('📑 Licitaciones')
st.sidebar.caption('5.021 licitaciones reales de Mercado Público (1-6 oct 2026)')
estados = st.sidebar.multiselect('Estado', sorted(df_all['estado'].unique()), default=sorted(df_all['estado'].unique()))
tramos = st.sidebar.multiselect('Tramo de monto', sorted(df_all['tramo'].unique()), default=sorted(df_all['tramo'].unique()))
filtro_txt = st.sidebar.text_input('Contiene la palabra (opcional)', '')
mask = df_all['estado'].isin(estados) & df_all['tramo'].isin(tramos)
if filtro_txt.strip():
    mask &= df_all['nombre'].fillna('').map(quitar_tildes).str.contains(quitar_tildes(filtro_txt.strip()), case=False)
df = df_all[mask].reset_index(drop=True)
E = E_all[mask.values]
st.sidebar.metric('Licitaciones seleccionadas', f'{len(df):,}'.replace(',', '.'))
st.sidebar.markdown('---')
st.sidebar.caption('Fuente: API pública de Mercado Público (ChileCompra). '
                   'Embeddings: `paraphrase-multilingual-MiniLM-L12-v2`.')

st.title('Text mining + Transformers sobre licitaciones públicas')
st.markdown('**1. Exploración clásica**: qué dice el texto, a simple vista. · '
            '**2. Transformers**: qué significa. · **3. Dashboard** de lo extraído.')

if len(df) < 30:
    st.warning('Hay muy pocas licitaciones con estos filtros. Amplía la selección.')
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(['1 · Exploración clásica', '2 · Transformers', '3 · Dashboard', 'Datos'])


@st.cache_data
def vectorizar(textos):
    cv = CountVectorizer(min_df=2)
    Xc = cv.fit_transform(textos)
    frec = pd.Series(np.asarray(Xc.sum(axis=0)).ravel(), index=cv.get_feature_names_out()).sort_values(ascending=False)
    tv = TfidfVectorizer(min_df=2)
    Xt = tv.fit_transform(textos)
    peso = pd.Series(np.asarray(Xt.mean(axis=0)).ravel(), index=tv.get_feature_names_out()).sort_values(ascending=False)
    return Xc, cv.get_feature_names_out(), frec, peso


textos = tuple(df['nombre_limpio'])
Xc, vocab, frec, peso_tfidf = vectorizar(textos)

# ------------------------------------------------------------------ 1. clásica
with tab1:
    st.subheader('Nube de palabras')
    st.caption('Términos frecuentes en los nombres de las licitaciones. Llamativa, pero no permite comparar magnitudes.')
    nube = WordCloud(width=1400, height=520, background_color='white', colormap='viridis', max_words=150,
                     random_state=42, collocations=False).generate(' '.join(textos))
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.imshow(nube, interpolation='bilinear'); ax.axis('off')
    st.pyplot(fig, width='stretch')

    st.subheader('Barras de frecuencia y TF-IDF')
    st.caption('La versión cuantitativa de la nube. TF-IDF penaliza las palabras que aparecen en casi todos los documentos.')
    k = st.slider('Cantidad de palabras', 10, 30, 20, key='k')
    c1, c2 = st.columns(2)
    with c1:
        f = frec.head(k)[::-1]
        st.plotly_chart(px.bar(x=f.values, y=f.index, orientation='h', title='Frecuencia',
                               labels={'x': 'veces', 'y': ''}, color_discrete_sequence=[AZUL]).update_layout(height=520),
                        width='stretch')
    with c2:
        p = peso_tfidf.head(k)[::-1]
        st.plotly_chart(px.bar(x=p.values, y=p.index, orientation='h', title='TF-IDF promedio',
                               labels={'x': 'peso', 'y': ''}, color_discrete_sequence=[NARANJO]).update_layout(height=520),
                        width='stretch')

    st.subheader('N-gramas')
    st.caption('Frases que se repiten, como "mantención preventiva" o "artículos de aseo".')
    n = st.radio('Tamaño', [2, 3], format_func=lambda x: 'Bigramas' if x == 2 else 'Trigramas', horizontal=True)
    vn = CountVectorizer(ngram_range=(n, n), min_df=2)
    Xn = vn.fit_transform(df['nombre_suave'])
    ng = pd.Series(np.asarray(Xn.sum(axis=0)).ravel(), index=vn.get_feature_names_out()).sort_values(ascending=False).head(15)[::-1]
    st.plotly_chart(px.bar(x=ng.values, y=ng.index, orientation='h', labels={'x': 'veces', 'y': ''},
                           color_discrete_sequence=[VERDE]).update_layout(height=460), width='stretch')

    st.subheader('Red de co-ocurrencia')
    st.caption('Qué palabras aparecen juntas en el mismo nombre. Grosor = veces que aparecen juntas.')
    top = set(frec.head(45).index)
    pares = collections.Counter()
    for t in textos:
        pares.update(itertools.combinations(sorted(set(t.split()) & top), 2))
    G = nx.Graph()
    for (a, b), w in pares.most_common(70):
        G.add_edge(a, b, weight=w)
    if G.number_of_edges():
        pos = nx.spring_layout(G, k=0.6, seed=42)
        wmax = max(d['weight'] for *_, d in G.edges(data=True))
        figr = go.Figure()
        for u, v, d in G.edges(data=True):
            figr.add_trace(go.Scatter(x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode='lines',
                                      line=dict(width=0.5 + 6 * d['weight'] / wmax, color='rgba(9,29,68,0.3)'),
                                      hoverinfo='text', text=f'{u} + {v}: {d["weight"]}', showlegend=False))
        nodos = list(G.nodes())
        figr.add_trace(go.Scatter(x=[pos[n_][0] for n_ in nodos], y=[pos[n_][1] for n_ in nodos], mode='markers+text',
                                  text=nodos, textposition='top center', hovertext=[f'{n_}: {frec[n_]}' for n_ in nodos],
                                  hoverinfo='text', marker=dict(size=[8 + 30 * frec[n_] / frec.max() for n_ in nodos],
                                                                color=NARANJO, line=dict(width=1, color='white')),
                                  showlegend=False))
        figr.update_layout(height=620, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figr, width='stretch')

    st.subheader('Temas con LDA')
    st.caption('LDA agrupa licitaciones por tema mirando solo los conteos de palabras.')
    n_temas = st.slider('Número de temas', 4, 12, 8)

    @st.cache_data
    def lda_temas(textos, n_temas):
        cv = CountVectorizer(min_df=2)
        X = cv.fit_transform(textos)
        lda = LatentDirichletAllocation(n_components=n_temas, random_state=42, learning_method='batch', max_iter=20)
        D = lda.fit_transform(X)
        v = cv.get_feature_names_out()
        palabras = [', '.join(v[i] for i in lda.components_[t].argsort()[::-1][:7]) for t in range(n_temas)]
        return D.argmax(axis=1), palabras

    asign, palabras = lda_temas(textos, n_temas)
    df['tema_lda'] = asign
    res = pd.DataFrame({'Tema': range(n_temas), 'Palabras clave': palabras,
                        'N.º licitaciones': np.bincount(asign, minlength=n_temas)})
    st.dataframe(res, hide_index=True, width='stretch')

# ------------------------------------------------------------------ 2. Transformers
with tab2:
    st.subheader('Mapa de embeddings (UMAP)')
    st.caption('Cada punto es una licitación. Las parecidas quedan juntas aunque usen palabras distintas. Pasa el mouse sobre los puntos.')
    color = st.radio('Colorear por', ['Tema BERTopic', 'Tema LDA', 'Estado', 'Tramo'], horizontal=True)
    col = {'Tema BERTopic': 'etiqueta_topico', 'Tema LDA': 'tema_lda', 'Estado': 'estado', 'Tramo': 'tramo'}[color]
    plot = df.copy()
    plot[col] = plot[col].astype(str)
    if color == 'Tema BERTopic':
        ocultar = st.checkbox('Ocultar licitaciones sin tema', value=True)
        if ocultar:
            plot = plot[plot['topico'] != -1]
    figm = px.scatter(plot, x='x', y='y', color=col, hover_name='nombre', hover_data={'x': False, 'y': False, 'estado': True},
                      opacity=0.75, height=640, labels={col: color})
    figm.update_traces(marker_size=5)
    figm.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), legend=dict(font=dict(size=10)))
    st.plotly_chart(figm, width='stretch')

    st.subheader('Clusters semánticos con BERTopic')
    st.caption('Temas obtenidos agrupando los embeddings (UMAP + HDBSCAN) y describiendo cada grupo con c-TF-IDF. '
               'Calculados sobre las 5.021 licitaciones.')
    tb = temas_bt[temas_bt['topico'] != -1].head(20)
    st.plotly_chart(px.bar(tb[::-1], x='n', y='etiqueta', orientation='h', labels={'n': 'N.º licitaciones', 'etiqueta': ''},
                           color_discrete_sequence=[CELESTE]).update_layout(height=620), width='stretch')

    st.subheader('Licitaciones similares: palabras vs. significado')
    st.caption('Elige una licitación y compara las más parecidas según TF-IDF (palabras en común) y según embeddings (significado).')
    opciones = df['nombre'].fillna('').tolist()
    defecto = next((i for i, t in enumerate(opciones) if re.search(r'mantenci.n.*(caldera|ascensor|bomba)', t, re.I)), 0)
    i = st.selectbox('Licitación de consulta', range(len(opciones)), index=defecto, format_func=lambda j: opciones[j][:120])

    tv = TfidfVectorizer(min_df=1).fit(textos)
    T = tv.transform(textos).toarray()
    T = T / np.clip(np.linalg.norm(T, axis=1, keepdims=True), 1e-9, None)

    def top_sim(M, i, k=8):
        s = M @ M[i]
        orden = [j for j in np.argsort(-s) if j != i][:k]
        return pd.DataFrame({'Licitación': df['nombre'].iloc[orden].values, 'Similitud': s[orden].round(3)})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Con TF-IDF (palabras)**')
        st.dataframe(top_sim(T, i), hide_index=True, width='stretch')
    with c2:
        st.markdown('**Con embeddings (Transformer)**')
        st.dataframe(top_sim(E, i), hide_index=True, width='stretch')

    st.info('**Mapa de atención:** requiere ejecutar el modelo Transformer completo; está en el notebook del curso '
            '(sección 2.5), que corre en Google Colab.')

# ------------------------------------------------------------------ 3. dashboard
with tab3:
    det = df[df['tiene_detalle']].copy()
    det['fecha_publicacion'] = pd.to_datetime(det['fecha_publicacion'], errors='coerce')
    det['fecha_cierre_dt'] = pd.to_datetime(det['fecha_cierre'], errors='coerce')
    det['dias_para_ofertar'] = (det['fecha_cierre_dt'] - det['fecha_publicacion']).dt.days
    txt_det = (det['descripcion'].fillna('') + ' ' + det['items_descripcion'].fillna('')).str.lower()
    det['plazo_meses'] = txt_det.str.extract(r'(\d{1,3})\s*mes')[0].astype(float)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Licitaciones', f'{len(df):,}'.replace(',', '.'))
    k2.metric('Publicadas (abiertas)', f'{(df["estado"] == "Publicada").sum():,}'.replace(',', '.'))
    k3.metric('Con detalle completo', len(det))
    k4.metric('Mediana días para ofertar', '—' if det['dias_para_ofertar'].dropna().empty else int(det['dias_para_ofertar'].median()))

    c1, c2 = st.columns(2)
    with c1:
        t = df['tramo'].value_counts()
        st.plotly_chart(px.bar(x=t.values, y=t.index, orientation='h', title='Licitaciones por tramo de monto',
                               labels={'x': 'N.º', 'y': ''}, color_discrete_sequence=[AZUL])
                        .update_layout(height=380, yaxis={'categoryorder': 'total ascending'}), width='stretch')
    with c2:
        e = df['estado'].value_counts()
        st.plotly_chart(px.pie(values=e.values, names=e.index, title='Estado de las licitaciones', hole=0.5,
                               color_discrete_sequence=[AZUL, NARANJO, VERDE, CELESTE, '#999999']).update_layout(height=380),
                        width='stretch')

    st.markdown(f'#### Licitaciones con detalle completo (muestra aleatoria de {len(det)})')
    if len(det):
        c1, c2 = st.columns(2)
        with c1:
            r = det['region'].fillna('').str.replace('Región ', '', regex=False).str.strip().replace('', np.nan).dropna().value_counts().head(12)
            st.plotly_chart(px.bar(x=r.values, y=r.index, orientation='h', title='Por región', labels={'x': 'N.º', 'y': ''},
                                   color_discrete_sequence=[NARANJO]).update_layout(height=420, yaxis={'categoryorder': 'total ascending'}),
                            width='stretch')
        with c2:
            st.plotly_chart(px.histogram(det, x='dias_para_ofertar', nbins=20, title='Días entre publicación y cierre',
                                         labels={'dias_para_ofertar': 'días'}, color_discrete_sequence=[VERDE]).update_layout(height=420),
                            width='stretch')
        montos = det[(det['monto_estimado'].fillna(0) > 0) & (det['moneda'] == 'CLP')]
        if len(montos):
            st.plotly_chart(px.box(montos, y='monto_estimado', points='all', hover_name='nombre', log_y=True,
                                   title=f'Monto estimado publicado (CLP, escala log) · {len(montos)} licitaciones',
                                   labels={'monto_estimado': 'CLP'}, color_discrete_sequence=[AZUL]).update_layout(height=460),
                            width='stretch')
        st.markdown('**Campos extraídos del texto con reglas** (plazo en meses mencionado en la descripción):')
        st.dataframe(det[['codigo', 'nombre', 'organismo', 'region', 'monto_estimado', 'dias_para_ofertar', 'plazo_meses']]
                     .sort_values('monto_estimado', ascending=False), hide_index=True, width='stretch')

# ------------------------------------------------------------------ datos
with tab4:
    st.dataframe(df[['codigo', 'nombre', 'estado', 'tramo', 'fecha_cierre', 'organismo', 'region', 'monto_estimado']],
                 hide_index=True, width='stretch')
    st.download_button('Descargar CSV filtrado', df.drop(columns=['nombre_limpio', 'nombre_suave']).to_csv(index=False).encode('utf-8'),
                       'licitaciones_filtradas.csv', 'text/csv')
