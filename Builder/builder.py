import pyc3dserver as c3d
itf = c3d.c3dserver(False)

import sys
import pandas as pd
import math
import numpy as np

def getData(file_name, number=0, close_other=True):#les données intéressantes du fichier c3d
    if close_other:#sait pas si ça marche
        ret=c3d.close_c3d(itf)
    ret = c3d.open_c3d(itf, file_name)

    n_frames = c3d.get_last_frame(itf)
    lab=c3d.get_marker_names(itf)
    data=c3d.get_dict_markers(itf)['DATA']['POS']
    
    ret=c3d.close_c3d(itf)
    
    #Si toutes les noms des trajectoires commencent avec les memes caractères (ex: New), on supprime ces caractères redondants
    for i in range(1, len(lab[0])):
        if all(list(t[0:i]==lab[0][0:i] for t in lab)):
            pass;
        else:
            i-=1
            lab=[e[i:] for e in lab]
            datak=[e[i:] for e in data.keys()]
            data=dict(zip(datak, list(data.values())))
            break;
        
    
    labdico = {}
    for i in lab:
        df = pd.DataFrame(data[i], columns=['x', 'y', 'z']).apply(pd.to_numeric)
        df = df[(df.x!=0) & (df.y!=0) & (df.z!=0)]
        if not df.empty:
            labdico[i]=df.copy()       

    return n_frames, list(labdico.keys()), labdico  
    

#vrai si le marqueur lab commence à la frame
def is_new(lab, frame):
    return labdico[lab].index[0]==frame

#vrai si le marqueur lab fini à la frame
def is_old(lab, frame):
    return labdico[lab].index[-1]==frame

#vrai si le marqueur lab existe à la frame
def is_present(lab, frame):
    return frame in lab.index

#liste les marqueurs présents à la frame
def markers_present(frame):
    mark=[]
    for i in labdico:
        if is_present(labdico[i], frame):
            mark.append(i)
    return mark
    
#retourne le nombre de marqueur présents à la frame
def n_markers_present(frame):
    n=0
    for i in labdico:
        if is_present(labdico[i], frame):
            n+=1
    return n
    
#vrai si 2 marqueurs (représentés par des dataframes) se superposent
def superpos(df1, df2):
    if (df1.index[-1]<df2.index[0]) or (df2.index[-1]<df1.index[0]):
        return False
    else: return True
    

def avg_speed(traj, step):
    if len(labdico[traj])>step:
        cpt=0
        speed=0
        for frame in range(labdico[traj].index[0], labdico[traj].index[-1]-step, step):
            pos_future=labdico[traj].loc[frame+step]
            speed+=math.dist(labdico[traj].loc[frame], pos_future)
            cpt+=1
        return speed/(cpt*step)
    else:
        return 'step error'

def get_key_from_value(val):
    keys = [k for k, v in connect.items() if val in v]
    if keys:
        return keys[0]
    return None

#liste des tetes et queue du dictionaire connect
def get_head_tail_connect():
    if len(connect)==0:
        return [], []
    else:
        head=[]
        tail=[]
        for key, val in connect.items():
            head = [*head, val[0]]
            tail = [*tail, val[-1]]
        return head, tail

connect ={}
def connexion(lab1, lab2):#enrichie le dictionnaire connect en connectant les marqueurs à la chaine
    head, tail = get_head_tail_connect()
    if tail==None:
        connect[lab1] = [lab1, lab2]
    else:
        if (lab1 in tail) and (lab2 in head):
            key1=get_key_from_value(lab1)
            connect[key1]= connect[key1]+connect[lab2]
            del connect[lab2]
        elif lab1 in tail:
            key=get_key_from_value(lab1)
            connect[key].insert(len(connect[key]), lab2)
        elif lab2 in head:
            connect[lab2].insert(0, lab1)
            connect[lab1] = connect.pop(lab2)
        else :
            connect[lab1] = [lab1, lab2]

#===============================================================================================
#===============================================================================================
# Fonctions pour supprimer les artefacts:

def find_artifacts1(labdico, step, dist_max):#Supprime les marqueurs loin des autres en regardant toute les "artif_detec_step1" frames
    
    artef=[]
    frames_seen=[]
    
    for frame in range(step ,n_frames-step, step):
        mark_pres=markers_present(frame)
        #On dégage les marqueurs distants:
        for i in mark_pres:
            distant=True
            for j in mark_pres:
                if i!=j:
                    if math.dist(labdico[i].loc[frame], labdico[j].loc[frame]) < dist_max:
                        distant=False
                        break
            if distant:
                artef.append(i)
                
    return sorted(list(dict.fromkeys(artef)))

#===============================================================================================
def find_artifacts2(labdico, dist_max):#Supprime les marqueurs si loins des autres a leur naissance et mort
    
    artef=[]
    frames_seen=[]
    
    for r in [0, -1]:#0 pour la première frame, -1 pour la dernière
        frames_seen=[]
        for h in labdico:
            if not labdico[h].index[r] in frames_seen: #si n'a pas été vu:
                frames_seen.append(labdico[h].index[r]) #on l'ajoute à la liste, frame_seen[-1] devient la frame intéressante
                mark_pres = markers_present(frames_seen[-1]) #les mark qui sont là dans la frame
                mark=[]


                for i in mark_pres:
                    if r==0 and is_new(i, frames_seen[-1]):
                        mark.append(i) #les mark qui apparaissent avec la frame
                    elif r==-1 and is_old(i, frames_seen[-1]):
                        mark.append(i) #les mark qui disparaissent avec la frame

                #Les marqueurs qui apparaissent ou disparaissent sont ils loins des autres marqueurs présents à la frame d'apparition/de disparition ?
                for i in mark:
                    distant=True
                    for j in mark_pres:
                        if i!=j:
                            if math.dist(labdico[i].iloc[r], labdico[j].loc[frames_seen[-1]]) < dist_max:
                                distant=False
                                break                   
                    if distant:                    
                        artef.append(i)
            
    return sorted(list(dict.fromkeys(artef)))
    
#===============================================================================================
def find_artifacts3(labdico, step, dist_motion):#Supprime les marqueurs si ne bougent pas assez
    
    artef=[]
    
    for i in labdico:
        if len(labdico[i])>step:
            motion=False
            for frame in range(labdico[i].index[0], labdico[i].index[-1]-step, step):
                pos_future=labdico[i].loc[frame+step]
                if math.dist(labdico[i].loc[frame], pos_future) > dist_motion:
                    motion=True
                    break
            if not motion:
                artef.append(i)
                
    return sorted(list(dict.fromkeys(artef)))

#===============================================================================================
def find_artifacts4(labdico, step, dist_motion):#Supprime les marqueurs si ne bougent pas assez
    
    artef=[]
    
    for i in labdico:
        if len(labdico[i])>step:
            if avg_speed(i, step)<dist_motion:
                artef.append(i)
                
    return sorted(list(dict.fromkeys(artef)))

#===============================================================================================
def find_artifacts5(labdico, n_marqueurs):
    
    end_dict={} #une clé de end_dict est le nom d'une trajectoire, sa valeur est l'entier correspondant à sa dernière frame
    for i in labdico:
        end_dict[i]=labdico[i].index[-1]
    end_dict = dict(sorted(end_dict.items(), key=lambda item: item[1]))# les trajectoires qui se terminent en premier sont les premieres du dictionnaire 

    proba_artef={}
    end_fr=0
    for i in labdico:
        frame_start=labdico[i].index[0]# on regarde à l'apparition de chaque trajectoire si on a plus de 15 marqueurs
        if n_markers_present(frame_start)>n_marqueurs and frame_start>end_fr: #et après end_fr pour pas refaire les memes choses
            for j in end_dict:# on cherche j: le marqueur qui a sa disparition fait qu'il n'y a plus plus de 15 marqueurs
                if end_dict[j]>frame_start and n_markers_present(end_dict[j])<=15:
                    for k in markers_present(frame_start): #on ajoute chaque marqueurs présents à proba_artef
                        if not k in proba_artef:
                            proba_artef[k]=0

                        if len(labdico[k])>200:
                            step=100
                        else:
                            step=10

                        proba_artef[k]+=avg_speed(k, step)

                    end_fr=end_dict[j]
                    break;#quitte la boucle de end_dict car on vient de trouver le end


    if proba_artef: #si le dico n'est pas vide
        return list({k: v for k, v in sorted(proba_artef.items(), key=lambda item: item[1])}.keys())[0]
    else:
        return []
        
#==================================================================================================
def delete_artifacts(lab, labdico):
    
    artifacts = list(dict.fromkeys(find_artifacts3(labdico, 75, 3.5)+find_artifacts4(labdico, 80, 0.01625)))
    lab=[x for x in lab if x not in set(artifacts)]
    for i in artifacts:
        del labdico[i]

    artif = artifacts
    artifacts = []
    
    while True:
        artifacts=list(dict.fromkeys(find_artifacts1(labdico, 90, 670)+find_artifacts2(labdico, 625)))
        artif += artifacts
        if artifacts == []:
            break;
        lab=[x for x in lab if x not in set(artifacts)]
        for i in artifacts:
            del labdico[i]

    while True:
        artifacts=find_artifacts5(labdico, 15)
        if artifacts == []:
            break;
        else:
            lab=[x for x in lab if x not in set(artifacts)]
            del labdico[artifacts]#artifact a une taille de 1, donc pas besoin de faire de boucle
            artif += [artifacts]

    return sorted(artif)

#================================================================================================
#===============================================================================================
#Fonctions de nommage:

def nomination(lab, labdico):
    
    name_list=markers_present(0)

    #On trie les marqueurs suivant l'axe z
    namer_sort={}
    for i in name_list:
        namer_sort[i]=labdico[i].iloc[0].z
    namer_sort=dict(sorted(namer_sort.items(), key=lambda item: item[1]))

    #On nomme la tete
    tete = list(namer_sort.keys())[-1]
    lab = namer(tete, 'Tete', lab, labdico, namer_sort)
    namer_sort=dict(sorted(namer_sort.items(), key=lambda item: item[1]))
    
    #On cherche le torse
    torse_ep_save = list(namer_sort.keys())[:-1][-3:]
    torse_ep_dict={}
    for i in torse_ep_save[:-1]:
        for j in torse_ep_save[1:]:
            if i!=j:
                torse_ep=torse_ep_save.copy()
                torse_ep.remove(i)
                torse_ep.remove(j)
                torse_ep_dict[torse_ep[0]]=math.dist(labdico[i].iloc[0], labdico[j].iloc[0])
    torse_ep_dict_sort= dict(sorted(torse_ep_dict.items(), key=lambda item: item[1]))

    #On nomme le torse
    torse = list(torse_ep_dict_sort.keys())[-1]
    lab = namer(torse, 'Torse', lab, labdico, namer_sort)
    namer_sort=dict(sorted(namer_sort.items(), key=lambda item: item[1]))

    #On nomme le bassin
    bassin = list(namer_sort.keys())[-7]
    lab = namer(bassin, 'Bassin', lab, labdico, namer_sort)
    namer_sort=dict(sorted(namer_sort.items(), key=lambda item: item[1]))

    #On nomme les épaules
    ep = list(torse_ep_dict_sort.keys())[:-1]
    lab = namer_side(ep, 'ep', lab, labdico, namer_sort, True)
    namer_sort=dict(sorted(namer_sort.items(), key=lambda item: item[1]))
    
    #On nomme les coudes
    coude = list(namer_sort.keys())[9:11]
    lab = namer_side(coude, 'Coude', lab, namer_sort, labdico)
    namer_sort=dict(sorted(namer_sort.items(), key=lambda item: item[1]))

    #On nomme les pieds
    pieds = list(namer_sort.keys())[0:2]
    lab = namer_side(pieds, 'Pied', lab, namer_sort, labdico)
    namer_sort=dict(sorted(namer_sort.items(), key=lambda item: item[1]))

    #On nomme les genoux
    genoux = list(namer_sort.keys())[2:4]
    lab = namer_side(genoux, 'Genou', lab, namer_sort, labdico)
    namer_sort=dict(sorted(namer_sort.items(), key=lambda item: item[1]))

    #On cherche les mains
    main_hanche_save = list(namer_sort.keys())[4:8]
    main_hanche_dict={}
    for i in main_hanche_save:
        for j in main_hanche_save:
            if i!=j:
                main_hanche=main_hanche_save.copy()
                main_hanche.remove(i)
                main_hanche.remove(j)
                main_hanche_dict[i,j]=math.dist(labdico[i].iloc[0], labdico[j].iloc[0])
    main_hanche_dict_sort= dict(sorted(main_hanche_dict.items(), key=lambda item: item[1]))

    #On nomme les mains
    mains = list(list(main_hanche_dict_sort.keys())[-1])
    lab = namer_side(mains, 'Main', lab, labdico, namer_sort, True)
    namer_sort=dict(sorted(namer_sort.items(), key=lambda item: item[1]))

    #On nomme les hanches
    hanches = [item for item in main_hanche_save if item not in mains]
    lab = namer_side(hanches, 'Hanche', lab, labdico, namer_sort, True)
    namer_sort=dict(sorted(namer_sort.items(), key=lambda item: item[1]))
    
    
    return lab, labdico
    
    
def sider(marker):
    #J'utilise une formule mathématiques pour savoir si le marker est à gauche ou à droite de l'axe torse/bassin
    x1=labdico['Torse'].iloc[0].x
    y1=labdico['Torse'].iloc[0].y
    x2=labdico['Bassin'].iloc[0].x
    y2=labdico['Bassin'].iloc[0].y
    x=labdico[marker].iloc[0].x
    y=labdico[marker].iloc[0].y
    
    if (x-x1)*(y2-y1)-(y-y1)*(x2-x1)>0:
        return 'droit'
    else:
        return 'gauche'
    
def namer_side(mark, mark_str, lab, labdico, namer_sort, fem=False):
    for i in mark:
        name= mark_str+'_' + sider(i)
        if sider(i)=='droit' and fem:
            name=name+'e'
        lab = list(map(lambda x: x.replace(i, name), lab))
        labdico[name] = labdico.pop(i)
        namer_sort[name] = namer_sort.pop(i)
    return lab

def namer(mark, mark_str, lab, labdico, namer_sort):
    lab = list(map(lambda x: x.replace(mark, mark_str), lab))
    labdico[mark_str] = labdico.pop(mark)
    namer_sort[mark_str] = namer_sort.pop(mark)
    return lab


#================================================================================================================================================================================================
#===============================================================================================================================================================================================
for file in sys.argv[1:]:# pour chaque file en argument

    print('Traitement du fichier '+ file)
    
    #on récupère les données:
    n_frames, lab, labdico = getData(file)
    file=file[:-4]

    artifacts=delete_artifacts(lab, labdico)
    
    lab=list(labdico.keys())        
    
    print('Les artefacts ont été supprimés:', artifacts)
    
    if n_markers_present(0)==15:
        lab, labdico = nomination(lab, labdico)
        print('Les noms ont été donné')
    else:
        print("Il n'y a pas 15 marqueurs à la première frame, les noms n'ont pas pu être donné")
    
    #on créer la dataframe dist qui stocke les distances spatio-temporel entre marqueurs
    str1= "'" + "', '".join(lab) + "'"
    str2= "'" + "', '".join(lab) + "'"

    dist = pd.DataFrame()
    exec("dist = pd.DataFrame(index=["+ str1 +"], columns=["+ str2 +"])")

    for i in lab:
        for j in lab:
            if not superpos(labdico[i], labdico[j]):
                if labdico[i].index[0] < labdico[j].index[0]:
                    d_space = math.dist(labdico[i].iloc[-1], labdico[j].iloc[0])
                    d_temp = labdico[j].index[0]-labdico[i].index[-1]
                    dist[j][i] = d_space*d_temp

    dist=dist.apply(pd.to_numeric)


    connect ={}

    #si un marqueur est présent du début à la fin, on le met dans le dictionnaire et on le supprime de la matrice "dist"
    for i in lab:
        if 0==labdico[i].index[0] and n_frames==labdico[i].index[-1]+1:
            connect[i]=[i]
            dist=dist.drop(i)
            dist=dist.drop(i, axis=1)


    #On cherche la distance minimum dans la matrice, on connecte les marqueurs dans le dictionnaire et on les supprime de la matrice dist jusqu'a ce que la plus petite distance restante soit d_max
    d_max=dist.max(1).max()+1
    while (dist.min(1).min()<d_max):
        min=dist.stack().idxmin()
        connexion(min[0], min[1])
        dist=dist.drop(min[0])
        dist=dist.drop(min[1], axis=1)

    #Création d'un fichier abstract.txt avec le dictionnaire connect et les artefacts détectés
    with open('Abstract.txt', "a") as o:
        o.write(str(file)+':\n')
        for k,v in sorted(connect.items()):
            o.write(str(k)+':')
            o.write(str(v))
            o.write('\n')
        o.write('Artefacts: '+str(artifacts)+'\n\n')

        print('Le fichier Abstract.txt a été mis à jour')

    fill_level = pd.DataFrame(columns=['Before', 'After'])


    

    #on liste les trajectoires de connect:
    liste=[]
    for k,v in connect.items():
        liste=liste+v

    #nombre de modification: nombre de marqueurs pas dans connect + taille de connect
    size=len(lab)-len(liste)+len(connect)

    cpt=0
    print('Marqueurs modifiés:')

    #On ouvre le fichier vide
    ret = c3d.open_c3d(itf, 'template.c3d')

    n_frames_temp=c3d.get_last_frame(itf)

    if n_frames_temp>n_frames:
        ret=c3d.delete_frames(itf, n_frames, n_frames_temp-n_frames)


    for key in labdico:
        if key in connect:
            val=connect[key]

            #Créer nouveau marqueur key:
            string="labdico['"+key+"']"
            key_size=len(labdico[key])
            chain_size=key_size
            for i in (val[1:]):
                string= string+", labdico['"+i+"']"
                chain_size=chain_size+len(labdico[i])
            exec("labdico['"+key+"']=pd.concat(["+string+"])")

            labdico[key]=labdico[key].loc[~labdico[key].index.duplicated(), :].sort_index()
            tab = labdico[key].reindex(list(range(0,n_frames)),fill_value=None)

            ret=c3d.add_marker(itf, key, tab.to_numpy(), adjust_params=True)
            #ret=c3d.fill_marker_gap_interp(itf, key, k=3, search_span_offset=5, min_needed_frs=10)#si il y a des trous

            cpt+=1
            print(str(cpt)+'/'+str(size)+':', val)

    ret=c3d.save_c3d(itf, f_path=file+'_build.c3d', compress_param_blocks=True)

    print('Le fichier '+str(file)+'_build a été créé\n') 

    ret=c3d.close_c3d(itf)

print('Tous les fichiers ont été traités\n') 
