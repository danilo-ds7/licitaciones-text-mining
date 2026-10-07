import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from texto_utils import limpiar
import umap
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
out=sys.argv[1]
df=pd.read_csv(f'{out}/licitaciones.csv'); V=np.load(f'{out}/embeddings.npy').astype(np.float32)
docs=df['nombre'].fillna('').tolist(); docs_l=[limpiar(d) for d in docs]
X2=umap.UMAP(n_neighbors=15,n_components=2,min_dist=0.08,metric='cosine',random_state=42).fit_transform(V)
X5=umap.UMAP(n_neighbors=15,n_components=5,min_dist=0.0,metric='cosine',random_state=42).fit_transform(V)
tm=BERTopic(umap_model=BaseDimensionalityReduction(), hdbscan_model=HDBSCAN(min_cluster_size=25,min_samples=10,prediction_data=True),
            vectorizer_model=CountVectorizer(ngram_range=(1,2),min_df=3), language='multilingual', calculate_probabilities=False)
topics,_=tm.fit_transform(docs_l, embeddings=X5)
info=tm.get_topic_info()
lab={r.Topic:('Sin tema' if r.Topic==-1 else ', '.join([w for w,_ in tm.get_topic(r.Topic)][:4])) for r in info.itertuples()}
mapa=pd.DataFrame({'codigo':df['codigo'],'x':X2[:,0].round(4),'y':X2[:,1].round(4),'topico':topics})
mapa['etiqueta_topico']=mapa['topico'].map(lab)
mapa.to_csv(f'{out}/mapa_embeddings.csv',index=False)
t=info[['Topic','Count']].rename(columns={'Topic':'topico','Count':'n'}); t['etiqueta']=t['topico'].map(lab)
t.to_csv(f'{out}/topicos_bertopic.csv',index=False)
print(len(info)-1,'temas; sin tema:',(np.array(topics)==-1).sum()); print(t.head(15).to_string())
