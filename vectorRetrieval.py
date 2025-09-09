import sys
import scipy
from math import floor
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx 
from scipy.spatial import Delaunay
from scipy.cluster.vq import kmeans, vq
import sys
sys.setrecursionlimit(30000000)

def euclidiana(u,v):
    return np.linalg.norm(u-v)

def cs(u,v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

class Nodo:
    def __init__(self, vectores=None,padre=None, mediana=None, beta=None, unitario=None, prof=0, izq=None, der=None,*,visitado=False):
        self.vectores = vectores
        self.mediana = mediana
        self.beta = beta
        self.unitario = unitario
        self.padre = padre
        self.prof = prof
        self.izq = izq
        self.der = der
        self.visitado = visitado

    def es_hoja(self):
        return self.vectores is not None
    

def construye_arbol_medianas(base, m0, prof):
    # Aquí escribimos el caso base de la funcion, devolviendo una hoja si se cumple alguna condicion que se especifica en el pseudocódigo proporcionado
    if base.shape[0] <= m0 or base.shape[1] == prof:
        # Se devuelve la clase mayoritaria y la distribución con un diccionario que asocia cada clase al número de elementos que tiene asociada
        return Nodo(vectores=base, prof=prof)
    else:
        coordenada = base[:, prof]
        mediana = np.median(coordenada)
        # Dividimos la base
        base_izq = base[coordenada < mediana]
        base_der = base[coordenada >= mediana]

        # Y hacemos las llamadas recursivas
        A_izq = construye_arbol_medianas(base_izq, m0, prof + 1)
        A_der = construye_arbol_medianas(base_der, m0, prof + 1)

        # Una vez terminadas, devolvemos el nodo con todo lo calculado en la función
        return Nodo(mediana=mediana, prof=prof, izq=A_izq, der=A_der)

def construye_arbol_random(base, m0, prof):
    # Aquí escribimos el caso base de la funcion, devolviendo una hoja si se cumple alguna condicion que se especifica en el pseudocódigo proporcionado
    if base.shape[0] <= m0 or base.shape[1] == prof:
        # Se devuelve la clase mayoritaria y la distribución con un diccionario que asocia cada clase al número de elementos que tiene asociada
        return Nodo(vectores=base, prof=prof)
    else:
        vector = np.random.randn(base.shape[1])
        unitario = vector / np.linalg.norm(vector)
        beta = 100 * (np.random.rand() * (1 / 2)) + 1 / 4
        escalares = np.dot(unitario, base.T)
        proyecciones = escalares / np.linalg.norm(base)
        percentil = np.percentile(proyecciones, beta)
        # Dividimos la base
        base_izq = base[escalares < percentil]
        base_der = base[escalares >= percentil]

        # Y hacemos las llamadas recursivas
        A_izq = construye_arbol_random(base_izq, m0, prof + 1)
        A_der = construye_arbol_random(base_der, m0, prof + 1)

        # Una vez terminadas, devolvemos el nodo con todo lo calculado en la función
        return Nodo(beta=beta, unitario=unitario, prof=prof, izq=A_izq, der=A_der)
    
def construye_arbol_spill(base, m0, prof, alfa):
    # Aquí escribimos el caso base de la funcion, devolviendo una hoja si se cumple alguna condicion que se especifica en el pseudocódigo proporcionado
    if base.shape[0] <= m0 or base.shape[1] == prof:
        # Se devuelve la clase mayoritaria y la distribución con un diccionario que asocia cada clase al número de elementos que tiene asociada
        return Nodo(vectores=base, prof=prof)
    else:
        vector = np.random.randn(base.shape[1])
        unitario = vector / np.linalg.norm(vector)
        escalares = np.dot(unitario, base.T)
        proyecciones = escalares / np.linalg.norm(base)
        mediana = np.median(proyecciones)
        (izq, der) = (np.percentile(proyecciones, 100*(1/2 - alfa)), np.percentile(proyecciones, 100*(1/2 + alfa)))
        # Dividimos la base
        base_izq = base[proyecciones < der]
        base_der = base[proyecciones >= izq]
        # Y hacemos las llamadas recursivas
        A_izq = construye_arbol_spill(base_izq, m0, prof + 1, alfa)
        A_der = construye_arbol_spill(base_der, m0, prof + 1, alfa)

        # Una vez terminadas, devolvemos el nodo con todo lo calculado en la función
        return Nodo(mediana=mediana, unitario=unitario, prof=prof, izq=A_izq, der=A_der)
    
class BranchCut:
    def __init__(self, vectors, m0):
        self.m0 = m0
        self.vectors = vectors # en formato array bidimensional de numpy
        self.kdTree = construye_arbol_medianas(self.vectors, m0, 0)
        self.betaTree = construye_arbol_random(self.vectors, m0, 0)
        #self.spillTree = construye_arbol_spill(self.vectors, m0, 0, 0.2)

    def search(self, query, random=False):
        nodo = self.kdTree
        nodo_beta = self.betaTree
        #nodo_spill = self.spillTree
        if random:
        #     opcion = input("Escribe el tipo de busqueda que quieras realizar (beta / spill)")
        #     if opcion == "beta":
            return self.search_derrotista(query, nodo_beta)
        #     elif opcion == "spill":
        #         return self.search_derrotista(query, nodo_spill)
        #     else:
        #         return "Opcion de busqueda incorrecta"
        return self.search_kd(query, nodo, [], None,0)

    def search_kd(self, q, nodo, recorrido, opt,m):
        if nodo.es_hoja():
            distancias = [euclidiana(q,v) for v in nodo.vectores]
            if not distancias:
                nodo.visitado = True
                padre = recorrido[-1]
                return self.search_kd(q, padre, recorrido[:-1], opt,m + 1)
            min_local = (nodo.vectores[np.argmin(distancias)], np.min(distancias))
            if opt is None or opt[1] > min_local[1]:
                opt = min_local
            nodo.visitado = True
            padre = recorrido[-1]
            return self.search_kd(q, padre, recorrido[:-1], opt,m + 1)
        else:
            if not nodo.izq.visitado and not nodo.der.visitado:
                recorrido.append(nodo)
                index = nodo.prof
                comprueba_opt = opt is None or np.abs(q[nodo.prof] - nodo.mediana) < opt[1] 
                if comprueba_opt:
                    if (q[index] < nodo.mediana):
                        print(m)
                        return self.search_kd(q, nodo.izq, recorrido, opt,m + 1)
                    else:
                        print(m)
                        return self.search_kd(q, nodo.der, recorrido, opt,m + 1)
                else:
                    nodo.visitado = True
                    padre = recorrido[-1]
                    print(m)
                    return self.search_kd(q, padre, recorrido[:-1], opt, m + 1)
            else:
                if not nodo.izq.visitado and np.abs(q[nodo.prof] - nodo.mediana) < opt[1]:
                    recorrido.append(nodo)
                    print(m)
                    return self.search_kd(q, nodo.izq, recorrido, opt,m + 1)
                elif not nodo.der.visitado and np.abs(q[nodo.prof] - nodo.mediana) < opt[1]:
                    recorrido.append(nodo)
                    print(m)
                    return self.search_kd(q, nodo.der, recorrido, opt,m + 1)
                else:
                    if not recorrido:
                        return opt, m
                    nodo.visitado = True
                    padre = recorrido[-1]
                    print(m)
                    return self.search_kd(q, padre, recorrido[:-1], opt, m + 1)

    def search_derrotista(self, query, nodo):
        if nodo.es_hoja():
            distancias = [euclidiana(query,v) for v in nodo.vectores]
            min_local = (self.vectors[np.argmin(distancias)], np.min(distancias))
            return min_local
        else:
            proyeccion = np.dot(nodo.unitario, query.T) / np.linalg.norm(q)
            if nodo.beta:
                if proyeccion < nodo.beta:
                    return self.search_derrotista(query, nodo.izq)
                else:
                    return self.search_derrotista(query, nodo.der)
            else:
                if proyeccion < nodo.mediana:
                    return self.search_derrotista(query, nodo.izq)
                else:
                    return self.search_derrotista(query, nodo.der)
                
def imprimir_arbol(nodo, nivel=0):
    if nodo is None:
        return
    print("  " * nivel + f"Nodo (prof: {nodo.prof}, unitario: {nodo.unitario}, vectores: {nodo.vectores}, mediana: {nodo.mediana})")
    imprimir_arbol(nodo.izq, nivel + 1)
    imprimir_arbol(nodo.der, nivel + 1)

class CoverTree:
    def __init__(self, vector, nivel, hijos, padre = None):
        self.vector = vector
        self.nivel = nivel
        self.padre  = padre
        self.hijos = hijos
    
    def igual(self, cover):
        return np.all(self.vector == cover.vector)
    
    def imprimir(self):
        print("vector:" + str(self.vector))
        print("nivel: " + str(self.nivel))
        print("hijos: " + str(self.hijos))

def inserta(p, v):
    if euclidiana(p.vector, v) > 2 ** p.nivel:
        while euclidiana(p.vector, v) > 4 ** p.nivel:
            q = p.children.pop()
            p_prima = CoverTree(q.vector, q.nivel + 1, [p])
            p = p_prima
        raiz = CoverTree(v, p.level + 1)
        raiz.hijos.append(p)
        return raiz
    return insertaAux(p, v)

def insertaAux(p, v):
    for q in p.hijos:
        if euclidiana(q.vector, v) <= 2 ** p.nivel:
            q_prima = insertaAux(q, v)
            p.hijos = [q_prima if x.igual(q) else x for x in p.hijos]
            return p
    x = CoverTree(v, p.nivel - 1, [])
    p.hijos.append(x)
    return p

def maxdist (cover): 
    if len(cover.hijos) > 0:
        return max([euclidiana(cover.vector, h.vector) for h in cover.hijos])  
    else:
        return 0

def buscarVecino(p, q, y = None):
    if y is None: 
        y = p.vector
    for c in sorted(p.hijos, key = lambda t : euclidiana(q, t.vector)):
        if euclidiana(y,q) > euclidiana(y, c.vector) - maxdist(c):
            y = buscarVecino(c, q, y)
    return y

def create_hyperplane_lsh(base, g):
    hashes = []
    signo = lambda v, h : np.dot(v, h) / np.abs(np.dot(v, h))
    buckets = {}
    for i in range(g):
        hashes.append(lambda v : signo(v, np.random.rand(base.shape[1])))
    for b in base:
        bk = str([hashes[n](b) for n in range(g)])
        print(bk)
        if bk not in buckets.keys():
            buckets[bk] = [b]
        else:
            buckets[bk].append(b)
    return hashes, buckets

def cross_politope_lsh(base, g):
    hashes = []
    print(base.shape[1])
    canon = np.identity(base.shape[1])
    rot = lambda v, R : canon[np.argmin([e - np.dot(R, v) / np.linalg.norm(np.dot(R, v)) for e in canon]) - 1]
    buckets = {}
    for i in range(g):
        hashes.append(lambda v : rot(v, np.random.normal(0,1,(base.shape[1], base.shape[1]))))
    for b in base:
        bk = str([hashes[n](b) for n in range(g)])
        if bk not in buckets.keys():
            buckets[bk] = [b]
        else:
            buckets[bk].append(b)
    return hashes, buckets

def alfabeta_lsh(base, g, r):
    hashes = []
    func = lambda v, alfa, beta : np.floor((np.dot(alfa, v) + beta) / r)
    buckets = {}
    for i in range(g):
        hashes.append(lambda v : func(v, np.random.normal(0, 1, base.shape[1]), np.random.randint(r)))
    for b in base:
        bk = str([hashes[n](b) for n in range(g)])
        if bk not in buckets.keys():
            buckets[bk] = [b]
        else:
            buckets[bk].append(b)
    return hashes, buckets

def search_LSH(q, buckets, hashes):
    qbk = str([h(q) for h in hashes])
    zona = buckets[qbk]
    distancias = [euclidiana(q,z) for z in zona]
    return zona[np.argmin(distancias)], np.min(distancias)

def voraz(G, q, k,s):
    Q = [s]
    visitados = [] # para implementar Vamana
    while True:
        S = set()
        for u in Q:
            S = S | set(G.neighbors(u))
        v = min([(n, euclidiana(q, attr['vector'])) for n, attr in G.nodes(data=True) if n in S], key = lambda x: x[1])[0]
        if v == Q[-1]:
            return Q, visitados
        Q.append(v) # si se incluye un elemento que ya este contenido en la lista, se elimina para terminar el algoritmo
        visitados.append(v)
        if len(Q) > k:
            Q.pop(0)

def GrafoDelaunay(base):
    D = nx.Graph()
    triangulacion = Delaunay(base)
    for i, u in enumerate(base):
        D.add_node(i, vector = u)
    for t in triangulacion.simplices:
        edges = [(t[i], t[j]) for i in range(3) for j in range(i+1, 3)]
        D.add_edges_from(edges)
    return D

def kNNGraph(base, k):
    G = nx.DiGraph()
    for i, b in enumerate(base):
        G.add_node(i, vector = b)
        dist = [euclidiana(b, u) for u in base]
        nn = [(i, j) for j in np.argsort(dist)[2:k + 1]]
        G.add_edges_from(list(nn))
    return G

def P_delaunay(D, base):
    # Probabilistic model that appends long distance edges
    delta_u = np.max([euclidiana(u, v) for u in base for v in base])
    delta_l = np.min([euclidiana(u, v) for u in base for v in base if euclidiana(u,v) > 0])
    for i, attr in D.nodes(data=True):
        u = attr.get('vector')
        alpha = np.random.uniform(np.log(delta_l), np.log(delta_u))
        theta = np.random.uniform(0, 2*np.pi)

        z = np.array([np.exp(alpha) * np.cos(theta), np.exp(alpha) * np.sin(theta)])
        u_prima = u + z
        
        j = min([(k, euclidiana(u_prima, attr2['vector'])) for k, attr2 in D.nodes(data=True) if euclidiana(u, attr2['vector']) > 0], key = lambda x : x[1])[0]
        D.add_edge(i, j)
    return D

def NSW(base, k):
    D = nx.complete_graph(k)
    for i, b in enumerate(base[:k]):
        D.nodes[i]['vector'] = b
    for i, b in enumerate(base[k:]):
        D.add_node(i + k, vector = b)
        distancias = [euclidiana(b, attr['vector']) for _, attr in D.nodes(data=True) if euclidiana(b, attr['vector'])>0]
        if(len(distancias) == k):
            aristas = [(i + k, j) for j in range(k)]
        else:
            aristas = [(i + k, j) for j in np.argpartition([euclidiana(b, attr['vector']) for _, attr in D.nodes(data=True) if euclidiana(b, attr['vector'])>0], k)[:k]]
        D.add_edges_from(aristas)
    return D 

def HeuristicEdges(grafo, nodo, candidatos, m):
    # Recibe un grafo sin aristas y las asigna siguiendo la heuristica del articulo del HNSW
    nAristas = [(nodo, candidatos[0])]
    for n in candidatos[1:]:
        dist = euclidiana(grafo.nodes()[nodo]['vector'], grafo.nodes()[n]['vector'])
        ma = list(map(lambda x : x[1], nAristas))
        comp = np.all([dist < euclidiana(grafo.nodes()[n]['vector'], grafo.nodes()[c]['vector'])  for c in ma])
        if comp:
            nAristas.append((nodo,n))
        if len(nAristas) == m: 
            break
    return nAristas

def HNSW(base, m, ml, efConstruction, heuristic = False):
    c = 0
    indices = enumerate(base)
    capas = {n: np.floor(-np.log(np.random.uniform(0,1)) * ml) for n, _ in indices}
    L = max(list(capas.values()))
    grafos = [nx.Graph() for _ in range(int(L))]
    for n, b in enumerate(base):
        for c in range(int(capas[n])):
            grafos[c].add_node(n, vector = b)
    for c in range(int(L)):
        nodos = grafos[c].nodes(data=True)
        for i, attr in nodos:
            candidatos = [j for j in 
                       list(map(lambda x : x[0],sorted([(j, euclidiana(attr['vector'], attr2['vector'])) 
                         for j, attr2 in nodos if euclidiana(attr['vector'], attr2['vector'])>0], 
                         key = lambda x : x[1])))[:efConstruction]]
            if candidatos:
                if heuristic:
                    aristas = HeuristicEdges(grafos[c], i, candidatos, m)
                else:
                    aristas = [(i,j) for j in candidatos[:m]]
                grafos[c].add_edges_from(aristas)
    return grafos, L, c
        
def vorazHNSW(hnsw, L, q):
    search = np.random.choice(hnsw[int(L - 1)].nodes())
    for l in range(int(L) - 1, 0, -1):
        if len(hnsw[l].nodes()) == 1:
            continue
        search = voraz(hnsw[l], q, 2, search)[0][1]
    return search, hnsw[0].nodes()[search]['vector']

def RNG(base):
    rng = nx.Graph()
    for n, b in enumerate(base):
        rng.add_node(n, vector = b)
    for i, u in enumerate(base):
        for j, v in enumerate(base):
            if euclidiana(u,v) > 0:
                comp = np.all([euclidiana(u, v) < max(euclidiana(u, w), euclidiana(v, w)) for w in base if euclidiana(u, w) > 0 and euclidiana(v, w) > 0])
                if comp:
                    rng.add_edge(i, j)
    return rng

def alphaSNG(base, alpha = 1):
    sng = nx.DiGraph()
    edges = []
    for i, u in enumerate(base):
        sng.add_node(i, vector = u)
        distancias = sorted([(j, v, euclidiana(u, v)) for j, v in enumerate(base)], key = lambda x : x[2])[2:]
        while distancias:
            j = distancias[0][0]
            v = distancias[0][1]
            edges.append((i, j))
            distancias = sorted([(j, w, d) for j, w, d in distancias if euclidiana(u, w) < alpha * euclidiana(v, w)], key = lambda x : x[2])
    sng.add_edges_from(edges)
    return sng

def Vamana(base, r, L, alpha):
    V = nx.random_regular_graph(d=r + 1, n=len(base))
    newEdges = []
    for i, b in enumerate(base):
        V.nodes[i]['vector'] = b
    for i, attr in V.nodes(data=True):
        u = attr['vector']
        visitados = voraz(V, u, L, np.random.randint(len(base)))[1]
        distancias = sorted([(j, attr['vector'], euclidiana(u, attr['vector'])) for j, attr in V.nodes(data=True) if j in visitados], key = lambda x : x[2])[2:]
        while distancias:
            j = distancias[0][0]
            v = distancias[0][1]
            newEdges.append((i, j))
            distancias = sorted([(j, w, d) for j, w, d in distancias if euclidiana(u, w) < alpha * euclidiana(v, w)], key = lambda x : x[2])
    V.add_edges_from(newEdges)
    return V

def clustering(base, c):    
    centroids, _ = kmeans(base, c)
    clusters, _ = vq(base, centroids)
    return centroids, clusters

def consultaCluster(base, centroids, clusters, q, k):
    centroide = np.argmin([euclidiana(q, c) for c in centroids])
    return base[np.argpartition([euclidiana(q, v) for i, v in enumerate(base) if clusters[i] == centroide], k)][:k]

def distribucionesMIPS(base):
    ptq = np.sum(base, axis=0)
    putq = []
    for u in base: 
        putq.append([u[t] / np.sum([base[:, t]]) for t in range(base.shape[1])])
    return ptq , np.array(putq)

def muestreo(base, ptq, putq, q):
    ptq = q * ptq
    nptq = ptq / np.sum(ptq)
    dim = np.random.choice(np.arange(q.shape[0]), p=list(nptq))
    vector = np.random.choice(np.arange(putq.shape[0]), p=list(putq[:, dim]))
    return base[vector]

def sMuestreos(base, ptq, putq, q, s, kprima, k):
    cont = {}
    for i in range(s):
        v = muestreo(base, ptq, putq, q)
        if tuple(v) not in cont:
            cont[tuple(v)] = 1
        else:
            cont[tuple(v)] +=1
    topkprima = sorted([v for v in cont],  reverse=True, key = lambda x : cont[x])[:kprima]
    topk = sorted([np.array(v) for v in topkprima], reverse=True, key = lambda x : np.dot(np.array(v), q))[:k]
    return topk

def boundedME(q, k, base, epsilon, delta):
    X = base
    m, d = X.shape
    A = np.zeros(m)
    t = 0
    
    h = lambda x : np.min([1 + x / 1 + (x/d), x + (x/d) / 1 + (x/d)])
    while len(X) > k:
        numerador = 2 * (len(X) - k)
        denominador = delta * (np.floor((len(X) - k) / 2) + 1)
        ti = h((2 / epsilon**2) * np.log(numerador / denominador))
        for u in X: 
            J = np.random.choice(d, int(ti - t), replace=False)
            A[J] = A[J] + u[J] * q[J]
        alpha = sorted(A)[int(np.floor((len(X) - k) / 2))]
        indices = np.where(A > alpha)
        X = X[indices]
        epsilon *= 3 / 4
        delta /= 2
    return X

def PQ(base, d0,c):
    d = base.shape[1]
    if d % d0 != 0:
        raise ValueError("La dimensión no es divisible entre d0")
    L = d // d0
    espacios = []
    for l in range(L):
        region = base[:,int(l*d0):int((l+1)*d0)]
        espacios.append(kmeans(region.astype(float),c)[0])
    return np.array(espacios)

def consultaPQ(espacios, q):
    d0 = espacios[0].shape[1]
    d = q.shape[0]
    L = d // d0
    resp = []
    for l in range(L):
        nc = espacios[l, np.argmin([euclidiana(e, q[int(l*d0):int((l+1)*d0)]) for e in espacios[l]])]
        for i in range(len(nc)):
            resp.append(nc[i])
    return np.array(resp)

def sketchingJL(base, d0):
    d = base.shape[1]
    phi = 1/np.sqrt(d0) * np.array([[1 if t > 0.5 else -1 for t in fila] for fila in  np.random.sample((d,d0))])
    return base @ phi