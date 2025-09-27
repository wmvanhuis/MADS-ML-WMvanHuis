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



Find the [notebook](./notebook.ipynb) and the [instructions](./instructions.md)

[Go back to Homepage](../README.md)
