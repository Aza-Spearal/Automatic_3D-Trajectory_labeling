import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import torch.optim as optim
import math
import random
import sys

import reader
from reader import read_data_csv
from reader import read_data_c3d

##tester le processeur graphique
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

####lire les données

file_c3d = sys.argv[1:]
if len(file_c3d) != 1:
        raise Exception('Il y a '+str(len(file_c3d))+ ' arguments donnés. Il faut mettre en argument un seul fichier c3d, voir la documentation') 
X_c3d = reader.read_data_c3d(file_c3d)
tensor_X_train, tensor_y_train, tensor_X_test, tensor_y_test, n_joints = reader.read_data_csv()

##définition des labels
LABELS = [     
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
        "Autre_Mouvement_Jambe"
    ]

##afficher la taille des ensembles d'apprentissage et test
n_data_size_test = len(tensor_X_test)
print('n_data_size_test:', n_data_size_test)

n_data_size_train = len(tensor_X_train)
print('n_data_size_train:', n_data_size_train)


##connaitre la classe de la sortie du modèle
def categoryFromOutput(output):
    return [output.topk(1)[1].item()]

##choix arbitraire de l'élément du batch à faire apprendre
def randomTrainingExampleBatch(batch_size, flag, num=-1):
    if flag == 'train':
        X = tensor_X_train
        y = tensor_y_train
        data_size = n_data_size_train
    elif flag == 'test':
        X = tensor_X_test
        y = tensor_y_test
        data_size = n_data_size_test
    if num == -1:
        ran_num = random.randint(0, data_size - batch_size)
    else:
        ran_num = num
        
    input_sequence = X[ran_num:(ran_num + batch_size)]
    category_tensor = y[ran_num:ran_num + batch_size]
    
    input_sequence = np.array(input_sequence)
    input_sequence = torch.from_numpy(input_sequence).float()
    input_sequence = input_sequence.to(device)
    
    category_tensor = np.array(category_tensor, dtype=int)
    category_tensor = torch.from_numpy(category_tensor - 1)
    category_tensor = category_tensor.to(device)
    
    return category_tensor, input_sequence

torch.backends.cudnn.enabled = False


##definition des paramétres du réseau
n_hidden = 32
n_categories = len(LABELS)
n_layer = 1

##Définition du modèle LSTM
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim #32
        self.output_dim = output_dim #6
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, layer_num, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        #self.soft_max = nn.Softmax(dim=1)

    def forward(self, inputs):
        lstm_out, (hn, cn) = self.lstm(inputs) #1,103, 32
        out = self.fc(lstm_out[:, -1, :]) #1, 6
        #out = self.soft_max(out)
        return out

rnn = LSTM(n_joints, n_hidden, n_categories, n_layer)
rnn.to(device)


learning_rate = 0.0005
optimizer = optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.9)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)


#### PHASE DE L'apprentissage
criterion = nn.CrossEntropyLoss()

batch_size = 1

# Keep track of losses for plotting
current_loss = 0
all_losses = []
n_iters = 14000
print_every = 100
plot_every = 100

print('\n', 'Epoch', 'Loss')
rnn.train()
for iter in range(1, n_iters + 1):
    
    es_param=10
    count=0

    category_tensor, input_sequence = randomTrainingExampleBatch(batch_size, 'train')
    
    if category_tensor.size()[1] != 1:
        category_tensor = torch.squeeze(category_tensor)
    else:#tout va la:
        category_tensor = category_tensor[0]
    optimizer.zero_grad()

    output = rnn(input_sequence)
    loss = criterion(output, category_tensor.long())
    loss.backward()
    optimizer.step()
    #scheduler.step()

    current_loss += loss.item()


    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        print(iter, loss.item())

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
    
plot1=plt.figure()
plt.plot(all_losses)



Y=[]
rnn.eval()
with torch.no_grad():
    input_tens=torch.Tensor(X_c3d[0][0])
    frame_beg=0
    
    with open('Results.txt', "a") as o:
        o.write('Debut  Fin  Label\n')
    print('Debut', 'Fin', 'Label')
    
    for i in range(len(input_tens)):
        input_seq=input_tens[i]
        input_seq=input_seq.unsqueeze(0).unsqueeze(0).float()
        input_seq=input_seq.to(device)
        output = rnn(input_seq)
        lab = categoryFromOutput(output)
        
        if i==0:
            actual_lab = lab
        elif lab != actual_lab or i == len(input_tens)-1: #quand changement de label ou fin du fichier, on affiche
            with open('Results.txt', "a") as o:
                o.write(str(frame_beg)+' '+str(i)+' '+str(actual_lab)+'\n')
            print(frame_beg, i, actual_lab)
            
            frame_beg = i
            actual_lab = lab
        
        Y.append(lab)