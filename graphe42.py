from os import truncate
import matplotlib.pyplot as plt
import networkx 
import numpy
import random
import time
import math

def creer_graphe(fname) : #générer un graphe non orienté à partir d'un fichier texte
    graphe = {}
    with open(fname,'r') as f :
        for line in f :
            lien = line.split(" ")
            if int(lien[0]) in graphe:
                graphe[int(lien[0])].append(int(lien[1]))
                if(int(lien[1]) in graphe):
                    graphe[int(lien[1])].append(int(lien[0]))
                else:
                    graphe[int(lien[1])] = [int(lien[0])]
            else :
                graphe[int(lien[0])] = [int(lien[1])]
    return graphe

def creer_graphe_oriente(fname): #générer un graphe orienté à partir d'un fichier texte
    graphe = {}
    with open(fname,'r') as f :
        for line in f :
            if line[0] == '%' :
                continue
            lien = line.split(" ")
            if int(lien[0]) in graphe :
                graphe[int(lien[0])][0].append(int(lien[1]))
            else :
                graphe[int(lien[0])] = [[int(lien[1])],[]]
            if int(lien[1]) in graphe :
                graphe[int(lien[1])][1].append(int(lien[0]))
            else :
                graphe[int(lien[1])]= [[],[int(lien[0])]]
    return graphe

def recherche_triades(grapheo): # algo de recherhce de triades dans un graphe orienté grapheo
    #grapheo[0] -> dico pour les successeurs graphe[1] -> prédecesseurs
    res = []
    for i,N in grapheo[0].items() :
        if N is not None:
            for j in N:
                if grapheo[0][j] is not None:
                    if i in grapheo[0][j]:
                        for k in grapheo[0][j]:
                            if grapheo[0][k] is not None:
                                if j in grapheo[0][k] and i in grapheo[0][k] and k in grapheo[0][i] :
                                    res.append((i,j,k))
    return res

def distribution_degree(graphe): #génère la distribution de degré d'un graphe
    """ retourne la distribution des degrees du graphe fourni en argument.
    Pour les graphes orientés, on ne fournit que sortant"""
    res = {}
    for edge,node in graphe.items() :
        if len(node) in res :
            res[len(node)] += 1 
        else :
            res[len(node)] = 1
    return res

def recherche_tri(graphe) : #algo n°1 de recherche de triangle dans un graphe non orienté (retourne une liste de triplet)
    res = set()
    for i,N in graphe.items():
        for j in N:
            if(j>i):
                if j in graphe:
                    for k in graphe[j]:
                        if k in N and k>j:
                            res.add((i,j,k))
    return res

def union_disjointe(l1,l2):
    L = l1 + l2
    return list(dict.fromkeys(L))

def tri_graphe(graphe): #trie par ordre croissant toutes les liste de voisins du graphe 
    for i,N in graphe.items():
        graphe[i] = sorted(N) #O(nlog(n))
    return graphe

def intersection_liste_triee(A,B):#retourne la liste intersection des listes A et B
    #A et B sont deux listes triees
    res = []
    i = 0
    j = 0

    while(i < len(A) and j<len(B)):
        if (A[i] < B[j]):
            i = i + 1
        elif B[j] < A[i]:
            j = j + 1
        else:
            res.append(A[i])
            i = i + 1
            j = j + 1
    return res

def recherche_tri_opti2(graphe): #algo n°2 de recherche de triangle dans un graphe 
    res = []
    #on cree seulement des 3-uplets croissant de triangle
    for i,N in graphe.items():
        for j in N: #pour chaque voisin du noeud i
            if j in graphe and j>i: #on verifie que j>i
                voisin_j = graphe[j]
                intersection = intersection_liste_triee(N,voisin_j)
                for k in intersection:
                    if(k>j):
                        res.append((i,j,k))
    return res

def recherche_tri_opti3(graphe):
    res = []
    marque = {}
    for i,N in graphe.items():
        if i not in marque:
            marque[i] = 0

    for i,N in graphe.items():
        for j in N: 
            marque[j] = 1

        for k in N:
            if(k>i):
                for l in graphe[k]:
                    if(l>k and marque[l]==1):
                        res.append((i,k,l))

        for j in N: 
            marque[j] = 0

    return res

def ER(n,m):
    graphe = dict()
    N = range(n)
    for i in N :
        graphe[i] = []
    i = 0

    while i < m :
        alea1 = random.randrange(n)
        alea2 = random.randrange(n)

        if (alea1 != alea2) and alea2 not in graphe[alea1] :
            graphe[alea1].append(alea2)
            graphe[alea2].append(alea1)
            i+=1

    return graphe 

def create_dist(n,e) : 
    liste = networkx.utils.random_sequence.powerlaw_sequence(n,exponent = e, seed = None)
    liste = [int(i) for i in liste]
    res = dict()
    for i in liste :
        if i in res:
            res[i]+=1
        else :
            res[i] = 1
    return res 

def config_model(dist): #genere un config model a partir d'une distribution de degree dist
    nom = 0
    tab = []
    graphe = dict() 
    for d,num in dist.items():
        for j in range(num):
            for k in range(d) :
                tab.append(nom)
            nom+=1
    #print(tab)
    i = len(tab)-1
    while i > 0 :
        temp1 = tab[i-1]
        temp2 = tab[i]
        alea1 = random.randint(0,i)
        alea2 = random.randint(0,i)
        tab[i-1] = tab[alea1]
        tab[i] = tab[alea2]
        tab[alea1] = temp1
        tab[alea2] = temp2
        i-=2

    #print(graphe)
    for k in range(0,len(tab)-1,2) :
        if tab[k] in graphe :
            graphe[tab[k]].append(tab[k+1])
        else :
            graphe[tab[k]] = [tab[k+1]]
        if tab[k+1] in graphe :
            graphe[tab[k+1]].append(tab[k])
        else :
            graphe[tab[k+1]] = [tab[k]]
    for edge,node in graphe.items() :
        #On elimine les doublons
        graphe[edge] = list(dict.fromkeys(node))
        if edge in node :
            #On elimine les boucles
            graphe[edge] = node.remove(edge)
     
        if type(graphe[edge]) is not list:
            graphe[edge] = []

    return graphe

def config_model_orientee(dist): #genere un graphe orienté à partir de deux distributions de degrés dist[0] et dist[1]
    nom = 0
    tab1 = []
    tab2 = []
    alea1 = 0

    dist1 = dist[0]
    dist2 = dist[1]
    graphe = [dict(),dict()]

    for d,num in dist1.items() :
        for j in range(num) :
            for k in range(d) :
                tab1.append(nom)
            nom+=1

    nom = 0

    for d,num in dist2.items() :
        for j in range(num) :
            for k in range(d) :
                tab2.append(nom)
            nom+=1
    #print(tab)
    i = len(tab1)-1

    for k in range(0,len(tab1)):
        alea1 = random.randint(0,len(tab2)-1)
        if tab1[k] in graphe[0]:
            graphe[0][tab1[k]].append(tab2[alea1])
        else:
            graphe[0][tab1[k]] = [tab2[alea1]]

    for k in range(0,len(tab2)):
        alea1 = random.randint(0,len(tab1)-1)
        if tab2[k] in graphe:
            graphe[1][tab2[k]].append(tab1[alea1])
        else:
            graphe[1][tab2[k]] = [tab1[alea1]]

    for edge,node in graphe[0].items() :
        #On elimine les doublons
        graphe[0][edge] = list(dict.fromkeys(node))
        if edge in node :
            #On elimine les boucles
            graphe[0][edge] = node.remove(edge)

    for edge,node in graphe[1].items() :
        #On elimine les doublons
        graphe[1][edge] = list(dict.fromkeys(node))
        if edge in node :
            #On elimine les boucles
            graphe[1][edge] = node.remove(edge)

    return graphe
    
def tracer_distribution(dist): #trace la distribution de degrés dist
    myList = dist.items()
    myList = sorted(myList) 
    print(myList)
    x,y = zip(*myList) 
    plt.scatter(x,y,s=15,c="red",marker = "o")
    plt.loglog(basex=10,basey=10)
    plt.xlabel("degré")
    plt.ylabel("nombre de noeuds")
    plt.title("Distribution de degré")
    plt.show()

def create_random_graphe(n):
    return config_model(create_dist(n,2.5))

def Z_score(graphe): #calcul le Z score d'un graphe par rapport à 100 graphes aléatoires
    Nreal = len(recherche_tri(graphe))
    Nliste = []
    somme = 0
    for i in range(100) :
        Nrandi = len(recherche_tri(create_random_graphe(len(graphe))))
        Nliste.append(Nrandi)
        somme += Nrandi
    Nmean = somme/len(Nliste)
    somme = 0
    for j in range(len(Nliste)) :
        Nliste[j] = abs(Nliste[j]-Nmean)**2
        somme += Nliste[j]
    Nstd = math.sqrt(somme/len(Nliste))
    return (Nreal - Nmean)/Nstd 

def SP(Zdict) : # prend un argument dico(nom_fichier:Zscore) et retourne un dictionnaire dico2(nom_fichier:SP)
    somme = 0
    for Z in Zdict :
        somme+= Zdict[Z]**2
    for name, Z in Zdict.items() :
        Zdict[name] = math.sqrt(Z/somme)
    return Zdict

def comparaison_graphe_reel(liste): #retourne le dictionnaire dico2(nom_fichier:SP) d'une liste de fichiers de graphes "liste"
    D = dict()
    for name in liste :
        g = creer_graphe(name)
        D[name] = Z_score(g)
    return SP(D)

def calcul_complexite_config_model_graphe(n) : #création graphique recherche triangles config model e = 2.5

    res1 = dict()
    res2 = dict()
    res3 = dict()
    N = range(0,n,100000)

    for i in N :
        dist = create_dist(i,2.5)
        graphe = config_model(dist)
        temp_graphe = graphe

        debut1 = time.perf_counter()
        recherche_tri(graphe)
        fin1 = time.perf_counter()

        tri_graphe(graphe)
        debut2 = time.perf_counter()
        recherche_tri_opti2(graphe)
        fin2 = time.perf_counter()

        debut3 = time.perf_counter()
        recherche_tri_opti3(temp_graphe)
        fin3 = time.perf_counter()

        res1[i] = (fin1-debut1)
        res2[i] = (fin2-debut2)
        res3[i] = (fin3-debut3)

        print(i)


    myList1 = res1.items()
    myList1 = sorted(myList1) 

    myList2 = res2.items()
    myList2 = sorted(myList2) 

    myList3 = res3.items()
    myList3 = sorted(myList3) 

    x1,y1 = zip(*myList1) 
    x2,y2 = zip(*myList2) 
    x3,y3 = zip(*myList3)

    plt.scatter(x1,y1,s=15,c="red",marker = "o")
    plt.scatter(x2,y2,s=15,c="blue",marker = "o")
    plt.scatter(x3,y3,s=15,c="green",marker = "o")

    mymodel1 = numpy.poly1d(numpy.polyfit(x1,y1,3))
    myline1 = numpy.linspace(0,n,100)

    mymodel2 = numpy.poly1d(numpy.polyfit(x2,y2,3))
    myline2 = numpy.linspace(0,n, 100)

    mymodel3 = numpy.poly1d(numpy.polyfit(x3,y3,3))
    myline3 = numpy.linspace(0,n,100)

    plt.plot(myline1,mymodel1(myline1),color = "red",label ='algo1')
    plt.plot(myline2,mymodel2(myline2),color = "blue",label ='algo2')
    plt.plot(myline3,mymodel3(myline3),color = "green", label ='algo3')
    plt.legend()
    plt.title("temps d'exécutions en fonction de n des algo1 et algo2 et algo3 avec e = 2.5")
    plt.xlabel("nombre de noeud n")
    plt.ylabel("temps(en s)")
    plt.show()

def calcul_complexite_graphe_dfixe(n,d): #création graphique recherche triangle ER(dfixe)

    res1 = dict()
    res2 = dict()
    res3 = dict()
    N = range(1,n,500000)

    for i in N :
        graphe = ER(i,int( (i*(i-1)*d) /2) )
        temp_graphe = graphe

        debut1 = time.perf_counter()
        recherche_tri(graphe)
        fin1 = time.perf_counter()

        tri_graphe(graphe)
        debut2 = time.perf_counter()
        recherche_tri_opti2(graphe)
        fin2 = time.perf_counter()

        debut3 = time.perf_counter()
        recherche_tri_opti3(temp_graphe)
        fin3 = time.perf_counter()

        res1[i] = (fin1-debut1)
        res2[i] = (fin2-debut2)
        res3[i] = (fin3-debut3)

        print(i)

    myList1 = res1.items()
    myList1 = sorted(myList1) 

    myList2 = res2.items()
    myList2 = sorted(myList2) 

    myList3 = res3.items()
    myList3 = sorted(myList3) 

    x1,y1 = zip(*myList1) 
    x2,y2 = zip(*myList2) 
    x3,y3 = zip(*myList3)

    plt.scatter(x1,y1,s=15,c="red",marker = "o")
    plt.scatter(x2,y2,s=15,c="blue",marker = "o")
    plt.scatter(x3,y3,s=15,c="green",marker = "o")

    mymodel1 = numpy.poly1d(numpy.polyfit(x1,y1,3))
    myline1 = numpy.linspace(0,n,100)

    mymodel2 = numpy.poly1d(numpy.polyfit(x2,y2,3))
    myline2 = numpy.linspace(0,n,100)

    mymodel3 = numpy.poly1d(numpy.polyfit(x3,y3,3))
    myline3 = numpy.linspace(0,n,100)

    plt.plot(myline1,mymodel1(myline1),color = "red",label = 'algo1')
    plt.plot(myline2,mymodel2(myline2),color = "blue",label = 'algo2')
    plt.plot(myline3,mymodel3(myline3),color = "green",label = 'algo3')
    plt.legend()

    plt.title("Temps d'exécutions en fonction de n des algo1/algo2/algo3 avec e = 2.5")
    plt.xlabel("Nombre de noeuds n")
    plt.ylabel("Temps(en s)")
    plt.show()

def calcul_complexite_graphe_nfixe(n,m): # création graphique recherche triangle ER(nfixe)

    res1 = dict()
    res2 = dict()
    res3 = dict()
    N = range(1,m,500000)

    for i in N :
        graphe = ER(n,i)
        temp_graphe = graphe

        debut1 = time.perf_counter()
        recherche_tri(graphe)
        fin1 = time.perf_counter()

        tri_graphe(graphe)
        debut2 = time.perf_counter()
        recherche_tri_opti2(graphe)
        fin2 = time.perf_counter()

        debut3 = time.perf_counter()
        recherche_tri_opti3(temp_graphe)
        fin3 = time.perf_counter()

        res1[i] = (fin1-debut1)
        res2[i] = (fin2-debut2)
        res3[i] = (fin3-debut3)
        print(i)

    myList1 = res1.items()
    myList1 = sorted(myList1) 

    myList2 = res2.items()
    myList2 = sorted(myList2) 

    myList3 = res3.items()
    myList3 = sorted(myList3) 

    x1,y1 = zip(*myList1) 
    x2,y2 = zip(*myList2) 
    x3,y3 = zip(*myList3)

    plt.scatter(x1,y1,s=15,c="red",marker = "o")
    plt.scatter(x2,y2,s=15,c="blue",marker = "o")
    plt.scatter(x3,y3,s=15,c="green",marker = "o")

    mymodel1 = numpy.poly1d(numpy.polyfit(x1,y1,3))
    myline1 = numpy.linspace(0,m,100)

    mymodel2 = numpy.poly1d(numpy.polyfit(x2,y2,3))
    myline2 = numpy.linspace(0,m,100)

    mymodel3 = numpy.poly1d(numpy.polyfit(x3,y3,3))
    myline3 = numpy.linspace(0,m,100)

    plt.plot(myline1,mymodel1(myline1),color = "red",label = 'algo1')
    plt.plot(myline2,mymodel2(myline2),color = "blue",label = 'algo2')
    plt.plot(myline3,mymodel3(myline3),color = "green",label = 'algo3')
    plt.legend()

    plt.title("Temps d'exécutions en fonction de m des algo1/algo2/algo3")
    plt.xlabel("Nombre de liens m")
    plt.ylabel("Temps(en s)")
    plt.show()
