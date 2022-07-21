import pyc3dserver as c3d
itf = c3d.c3dserver(False)

import sys
import pandas as pd
import math
import numpy as np

def data_to_df(labs, data):#transforme les données c3d en un dictionnaire de dataframe
    labdico = {}
    for i in range(len(labs)):
        labdico[labs[i]] = pd.DataFrame(data[labs[i]], columns=['x', 'y', 'z'])
        df = labdico[labs[i]].apply(pd.to_numeric)
        labdico[labs[i]] = df[(df.x!=0) & (df.y!=0) & (df.z!=0)]
    return labdico
    

def getData(file_name, close_other=True):#les données intéressantes du fichier c3d
    if close_other:#sait pas si ça marche
        ret=c3d.close_c3d(itf)
    ret = c3d.open_c3d(itf, file_name)

    n_frames = c3d.get_last_frame(itf)
    lab=c3d.get_marker_names(itf)
    data=c3d.get_dict_markers(itf)['DATA']['POS']
    
    ret=c3d.close_c3d(itf)
    
    labdico=data_to_df(lab, data)

    return n_frames, lab, data, labdico
    

def is_new(lab, frame):
    return labdico[lab].index[0]==frame

def is_old(lab, frame):
    return labdico[lab].index[-1]==frame

def is_present(lab, frame):
    return frame in lab.index

def markers_present(frame):
    mark=[]
    for i in labdico:
        if is_present(labdico[i], frame):
            mark.append(i)
    return mark
    
def superpos(df1, df2): #vrai si 2 marqueurs (représentés par des dataframes) se superposent
    if (df1.index[-1]<df2.index[0]) or (df2.index[-1]<df1.index[0]):
        return False
    else: return True

def get_key_from_value(val):
    keys = [k for k, v in connect.items() if val in v]
    if keys:
        return keys[0]
    return None

def get_head_tail_connect():#liste des tetes et queue du dictionaire connect
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
# 3  fonctions pour supprimer les artefacts:

def find_artifacts1(labdico, artif_detec_step1, artif_dist_max1):#Supprime les marqueurs loin des autres en regardant toute les "artif_detec_step1" frames
    dist_list=[]
    frames_seen=[]
    erase=[]
    erase_save=['']
    
    for frame in range(artif_detec_step1 ,n_frames-artif_detec_step1, artif_detec_step1):
        mark_pres=markers_present(frame)
        #On dégage les marqueurs distants:
        for i in mark_pres:
            distant=True
            for j in mark_pres:
                if i!=j:
                    if math.dist(labdico[i].loc[frame], labdico[j].loc[frame]) < artif_dist_max1:
                        distant=False
                        break
            if distant:
                dist_list.append(i)
                
    return sorted(list(dict.fromkeys(dist_list)))

#===============================================================================================
def find_artifacts2(labdico, artif_dist_max2):#Supprime les marqueurs si loins des autres a leur naissance et mort
    
    dist_list=[]
    frames_seen=[]
    erase=[]
    erase_save=['']
    
    while erase_save!=[]:#boucle peut etre inutile
        erase_save=[]
        for r in [0, -1]:
            frames_seen=[]
            for h in labdico:
                if not labdico[h].index[r] in frames_seen: #si n'a pas été vu:
                    frames_seen.append(labdico[h].index[r]) #on l'ajoute à la liste, frame_seen[-1] devient la frame intéressante
                    mark_pres = markers_present(frames_seen[-1]) #les mark qui sont là dans la frame
                    mark_new=[]
                    mark_old=[]
                    
                    if r==0:
                        for i in mark_pres:
                            if is_new(i, frames_seen[-1]):
                                mark_new.append(i) #les mark qui viennent d'apparaitre avec la frame
                    elif r==-1:
                        for i in mark_pres:
                            if is_old(i, frames_seen[-1]):
                                mark_old.append(i) #les mark qui disparaissent avec la frame

                    mark=mark_new+mark_old

                    for i in mark:
                        distant=True
                        for j in mark_pres:
                            if i!=j:
                                if math.dist(labdico[i].iloc[r], labdico[j].loc[frames_seen[-1]]) < artif_dist_max2:
                                    distant=False
                                    break                   
                        if distant:                    
                            erase_save.append(i)

            for i in erase_save:
                if i in labdico:
                    del labdico[i]

            erase=erase+erase_save  
            
        return sorted(list(dict.fromkeys(erase)))
    
    
#===============================================================================================
def find_artifacts3(labdico, artif_detec_step3, artif_dist_motion):#Supprime les marqueurs si ne bougent pas assez
    
    motion_list=[]
    for i in labdico:
        if len(labdico[i])>artif_detec_step3:
            motion=False
            for frame in range(labdico[i].index[0], labdico[i].index[-1]-artif_detec_step3, artif_detec_step3):
                i_future=labdico[i].loc[frame+artif_detec_step3]
                if math.dist(labdico[i].loc[frame], i_future) > artif_dist_motion:
                    motion=True
                    break
            if not motion:
                motion_list.append(i)
                
    return motion_list


#===============================================================================================
def find_artifacts(labdico):
    
    a=find_artifacts1(labdico, 100, 670)
    
    b=find_artifacts2(labdico, 625)
    
    c=find_artifacts3(labdico, 75, 3.5)
                
    return sorted(list(dict.fromkeys(a+b+c)))


#================================================================================================
#===============================================================================================
for file in sys.argv[1:]:# pour chaque file en argument

    print('Traitement du fichier '+str(file[:-4]))
    
    #on récupère les données:
    n_frames, lab, data, labdico = getData(file)
    file=file[:-4]

    #suppression des artefacts
    artifacts=find_artifacts(labdico)
    
    lab=[x for x in lab if x not in set(artifacts)]
    labdico=data_to_df(lab, data)
        
    print('Les artefacts ont été supprimés:', artifacts)

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
    d_max=60000 #dist.max(1).max()+1
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

            fill_level.loc[key] = [key_size/(n_frames+1), chain_size/(n_frames+1)]

            cpt+=1
            print(str(cpt)+'/'+str(size)+':', val)

    print('Marqueurs non modifiés:')
    liste=[item for item in lab if item not in liste]
    for key in liste:
        tab = labdico[key].reindex(list(range(0,n_frames)),fill_value=None)
        ret=c3d.add_marker(itf, key, tab.to_numpy(), adjust_params=True)
        cpt+=1
        print(str(cpt)+'/'+str(size)+':', key)
        
        
    fill_level['Gain'] = fill_level['After'] - fill_level['Before']
    print(fill_level.sort_values(by=['Gain'], ascending=False))

    ret=c3d.save_c3d(itf, f_path=file+'_build.c3d', compress_param_blocks=True)

    print('Le fichier '+str(file)+'_build a été créé\n') 

    ret=c3d.close_c3d(itf)
    
print('Tous les fichiers ont été traités\n') 