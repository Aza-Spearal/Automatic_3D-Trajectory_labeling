import os
import h5py
import numpy as np
import torch
import csv
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize


def joint_joint_distance(joints):
    D = pdist(joints)
    D = np.nan_to_num(D)
    return D

def joint_line_distance(joints,lines,mask):

    j1,j2 = joints[lines.swapaxes(1,0)]
    J1 = np.repeat(j1, 15, axis=0).reshape(25,15,3)
    J2 = np.repeat(j2, 15, axis=0).reshape(25,15,3)
    J3 = np.repeat(joints,25,axis=0).reshape(25,15,3,order='F')

    A = np.sqrt(np.sum(np.square(J1-J3),axis=2))
    B = np.sqrt(np.sum(np.square(J2-J3),axis=2))
    C = np.sqrt(np.sum(np.square(J1-J2),axis=2))

    P = (A+B+C)/2
    jl_d = 2*np.sqrt(P*(P-A)*(P-B)*(P-C))/(C)
    jl_d = np.ma.array(jl_d,mask=mask).compressed()
    jl_d = jl_d
    jl_d = np.nan_to_num(jl_d)
    return jl_d
    
    
lines = np.array([[12, 7], [2, 12], [7, 6], [6, 10],
              [1, 2], [1, 5], [9, 8], [3, 4], [9, 12],
              [4, 12]])

# J1 à l'extrémité, J2 est la deuxième jointure adjacente
lines = np.append(lines, [[11, 12], [2, 5], [7, 10],
                          [3, 12], [8, 12]], axis=0)

# J1 et J2 sont les deux à l'extrémité  
lines = np.append(lines, [[11, 10], [11, 5], [11, 8], [11, 3], [10, 5],
                          [3, 10], [8, 10], [5, 8], [3, 5], [3, 8]], axis=0)
lines = lines - 1

jl_mask = np.full((25, 15), False, dtype=bool)
for i in range(0, 25):
    jl_mask[i][lines[i]] = True
    
    
col_order = ['x_sq_Bassin', 'y_sq_Bassin', 'z_sq_Bassin', 'x_sq_D_coude', 'y_sq_D_coude', 'z_sq_D_coude', 'x_sq_D_epaule', 'y_sq_D_epaule', 'z_sq_D_epaule', 'x_sq_D_genou', 'y_sq_D_genou', 'z_sq_D_genou', 'x_sq_D_hanche', 'y_sq_D_hanche', 'z_sq_D_hanche', 'x_sq_D_main', 'y_sq_D_main', 'z_sq_D_main', 'x_sq_D_pied', 'y_sq_D_pied', 'z_sq_D_pied', 'x_sq_G_coude', 'y_sq_G_coude', 'z_sq_G_coude', 'x_sq_G_epaule', 'y_sq_G_epaule', 'z_sq_G_epaule', 'x_sq_G_genou', 'y_sq_G_genou', 'z_sq_G_genou', 'x_sq_G_hanche', 'y_sq_G_hanche', 'z_sq_G_hanche', 'x_sq_G_main', 'y_sq_G_main', 'z_sq_G_main', 'x_sq_G_pied', 'y_sq_G_pied', 'z_sq_G_pied', 'x_sq_Tete', 'y_sq_Tete', 'z_sq_Tete', 'x_sq_Torse', 'y_sq_Torse', 'z_sq_Torse']
    
    
def pos_to_JD(pos):
    frame_count = len(pos)
    ## pos contient les données brutes si on veut entrainer avec
    pos = np.array(pos).reshape(-1, 15, 3)
    ## JJ_D contient les distances entre jointures
    JJ_D = np.array(
        Parallel(n_jobs=24)(delayed(joint_joint_distance)(pos[i]) for i in range(frame_count)))

    ## JL_D contient les distances entre jointures et droites
    JL_D = np.array(
        Parallel(n_jobs=24)(
            delayed(joint_line_distance)(pos[i], lines, jl_mask) for i in range(frame_count)))

    ## JL_D et JJ_D concatinées
    return np.concatenate((JL_D, JJ_D), axis=1)




def read_data_csv():

    ##Spécification des classes
    classes = [
        "Assis_Debout",
        "Assis_Couche",
        "Couche_Assis",
        "Debout_Assis",
        "Debout_Agenou",
        "Agenou_Debout",
        "Debout_Penche",
        "Penche_Debout",
        "Autre_Transition",
        "Marcher",
        "Monter_Escaliers",
        "Descendre_Escaliers",
        "Lever_Bras",
        "Lever_2_Bras",
        "Baisser_Bras",
        "Baisser_2_Bras",
        "Autre_Mouvement_Bras",
        "Lever_Jambe",
        "Baisser_Jambe",
        "Autre_Mouvement_Jambe",
    ]

    ##calcul nombre des échantillants de train et test
    nb_seq_train = {
        "Assis_Debout":0,
        "Assis_Couche":0,
        "Couche_Assis":0,
        "Debout_Assis":0,
        "Debout_Agenou":0,
        "Agenou_Debout":0,
        "Debout_Penche":0,
        "Penche_Debout":0,
        "Autre_Transition":0,
        "Marcher":0,
        "Monter_Escaliers":0,
        "Descendre_Escaliers":0,
        "Lever_Bras":0,
        "Lever_2_Bras":0,
        "Baisser_Bras":0,
        "Baisser_2_Bras":0,
        "Autre_Mouvement_Bras":0,
        "Lever_Jambe":0,
        "Baisser_Jambe":0,
        "Autre_Mouvement_Jambe":0
    }

    nb_seq_test = {
        "Assis_Debout": 0,
        "Assis_Couche": 0,
        "Couche_Assis": 0,
        "Debout_Assis": 0,
        "Debout_Agenou": 0,
        "Agenou_Debout": 0,
        "Debout_Penche": 0,
        "Penche_Debout": 0,
        "Autre_Transition": 0,
        "Marcher": 0,
        "Monter_Escaliers": 0,
        "Descendre_Escaliers": 0,
        "Lever_Bras": 0,
        "Lever_2_Bras": 0,
        "Baisser_Bras": 0,
        "Baisser_2_Bras": 0,
        "Autre_Mouvement_Bras": 0,
        "Lever_Jambe": 0,
        "Baisser_Jambe": 0,
        "Autre_Mouvement_Jambe": 0
    }


    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    
    #files_train = ['AlaMhalla-22-06-29.csv', 'corinneMontlu-22-06-21.csv', 'didierMontlu-22-06-07.csv', 'didierMontlu-22-06-17.csv', 'EmmanuelBergeret-22-06-24.csv']
    #files_test = ['AlaMhalla-22-06-29.csv']
    files_train = []
    
    dirname = os.path.dirname(__file__)
    train_dir = os.path.join(dirname, 'csv_train')

    for root, dirs, files in os.walk(train_dir):
        for file in files:
        
            if not (file[:7] == '.~lock.'): #si le fichier csv est ouvert, un fichier qui commence par .~lock. est présent, on en veut pas
                with open(os.path.join(root, file), newline='') as csvfile:
                    reader = list(csv.reader(csvfile, delimiter=','))

                    if (reader[0][1:46] != col_order or reader[0][-1] != 'etiq activite'):
                        print("L'ordre ou le nom des colonnes n'est pas correct dans", file)
                    else:
                        print(file)
                        files_train.append(file)
                        i = 1
                        while i<len(reader):
                            row = reader[i]
                            frame_pos = [float(element) /1000 for element in row[1:46]]
                            label = row[-1]
                            if label == '':
                                i = i + 1
                                continue
                            else:
                                pos = []
                                while (row[-1] == label and i <= len(reader) - 1):
                                    row = reader[i]
                                    frame_pos = [float(element) /1000 for element in row[1:46]]
                                    pos.append(frame_pos)
                                    i = i + 1
         
                                listeconcat = pos_to_JD(pos)

                                if label in classes:
                                    y = [classes.index(label) + 1]
                                    print(y, end=' ')
                                    if file in files_train:
                                        nb_seq_test[label] = nb_seq_test[label] + 1
                                        X_train.append(listeconcat)
                                        Y_train.append(y)
                                    '''
                                    if file in files_test:
                                        nb_seq_train[label] = nb_seq_train[label] + 1
                                        X_test.append(listeconcat)
                                        Y_test.append(y)
                                    '''
    print(len(X_train))
    #print(len(X_test))
    #print(nb_seq_test)
    print(nb_seq_train)
    #listeconcat.shape[1] est la taille de l'input du NN
    return X_train, Y_train, X_test, Y_test, listeconcat.shape[1]


import builder
from builder import main
    
def read_data_c3d(file):

    n_frames, labdico = builder.main(file)
    
    marker_name = ['Bassin', 'D_coude', 'D_epaule', 'D_genou', 'D_hanche', 'D_main', 'D_pied', 'G_coude', 'G_epaule', 'G_genou', 'G_hanche', 'G_main', 'G_pied', 'Tete', 'Torse']
    
    for i in marker_name:
        if not i in labdico:
            raise Exception("Les noms n'ont pas pu êtres donnés, l'étiquettage est imposible")

    reader=[]

    reader.append(['x_sq_Bassin', 'y_sq_Bassin', 'z_sq_Bassin', 'x_sq_D_coude', 'y_sq_D_coude', 'z_sq_D_coude', 'x_sq_D_epaule', 'y_sq_D_epaule', 'z_sq_D_epaule', 'x_sq_D_genou', 'y_sq_D_genou', 'z_sq_D_genou', 'x_sq_D_hanche', 'y_sq_D_hanche', 'z_sq_D_hanche', 'x_sq_D_main', 'y_sq_D_main', 'z_sq_D_main', 'x_sq_D_pied', 'y_sq_D_pied', 'z_sq_D_pied', 'x_sq_G_coude', 'y_sq_G_coude', 'z_sq_G_coude', 'x_sq_G_epaule', 'y_sq_G_epaule', 'z_sq_G_epaule', 'x_sq_G_genou', 'y_sq_G_genou', 'z_sq_G_genou', 'x_sq_G_hanche', 'y_sq_G_hanche', 'z_sq_G_hanche', 'x_sq_G_main', 'y_sq_G_main', 'z_sq_G_main', 'x_sq_G_pied', 'y_sq_G_pied', 'z_sq_G_pied', 'x_sq_Tete', 'y_sq_Tete', 'z_sq_Tete', 'x_sq_Torse', 'y_sq_Torse', 'z_sq_Torse'])

    print('Conversion des '+ str(n_frames) +' frames pour le lstm')
    
    for i in range(0, n_frames):
        liste =[]
        for marker in marker_name:
            liste+=[labdico[marker].iloc[i].x, labdico[marker].iloc[i].y, labdico[marker].iloc[i].z]

        if i%1000 == 0:
            print(i, end=' ')
        reader.append(liste)

    
    X = []

    i = 1
    while i<len(reader):
        row = reader[i]
        frame_pos = [float(element) /1000 for element in row]
        pos = []
        while (i <= len(reader) - 1):
            row = reader[i]
            frame_pos = [float(element) /1000 for element in row]
            pos.append(frame_pos)
            i = i + 1        
        
        listeconcat = pos_to_JD(pos)
        
        X.append(listeconcat)
            
    print('\n')
    return X, listeconcat.shape[1]