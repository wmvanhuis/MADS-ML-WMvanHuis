# Summary week 3

Eerst heb ik de notebook stap voor stap gerund en MLflow werkende gekregen.
MLflow werkende krijgen ging nog niet helemaal zo makkelijk, maar is wel gelukt.

Daarna heb ik meerdere experimenten gedaan en de BaseRNN config meerdere keren aangepast.
Het verhogen van de hidden size leidt niet gelijk tot een betere accuracy is mij opgevallen.
Ik heb uiteindelijk met onderstaande settings >99% accuracy weten te behalen.
    input_size=3,
    hidden_size=256,    
    num_layers=2,       
    horizon=20,
    bidirectional=True,  
    dropout=0.3 

Nu met LSTM/GRU gewerkt.
Eerste run met LSTM gedaan.
zelfde input parameters gehouden als bij de RNN.
Accuracy nu >98%, dus iets lager dan met RNN.

Tweede run met GRU.
Zrelfde parameters weer
Accuracy nu >99%. Net zo goed als RNN.
Model was wel trager dan alle anderen.


Nu met CONV1Dlayers gewerkt.
Eerste experiment met LSTM.
inputparameters:
input_size    = 3
conv_channels = (32, 64)   
kernel_sizes  = (5, 3)     
pool          = 2          
hidden_size   = 128        
num_layers    = 1         
rnn_dropout   = 0.2
bidirectional = True
Model wordt trager met CONV1Dlayers.
Accuracy is nu ook >99%. achter de komma wel iets hoger dan de vorige.

Tweede experiment met GRU.
Zelfde inputparameters
Kreeg ik niet werkende. Kreeg een fout bij het script voor de accuracy.

Hypothese
Van te voren verwacht ik dat aanpassen van de parameters door het verogen van de hidden layers ook een verbetering zou worden. Dat is op een zekere hoogte ook zo, maar meer hidden layers betekent niet altijd betere accuracy. Het is zoeken naar de sweet spot.
Ik had verwacht dat met meer channels en meer kernels de accuracy zou toenemen. Het model wordt hierdoor complexer en dus ook trager. Accuracy nam toe.

Reflectie.
Ik begrijp wat de bedoeling is en wat er gebeurt.
Ik heb mijn code gemaakt met hulp van chatgpt, ik blijf het lastig vinden om dat zelf te doen, maar goed prompten, dus ook goed begrijpen, helpt wel met het zoeken naar de oplossing. 
Ik moet dit nog veel blijven oefenen vind ik zelf om het zoeken naar de optimale settings van het model en ook de modelkeuze beter te begrijpen.

Find the [notebook](./notebook.ipynb) and the [instructions](./instructions.md)

[Go back to Homepage](../README.md)
