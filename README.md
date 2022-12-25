# neural-network-CNN-for-classification


Il task di questo progetto è “classificare i differenti tocchi delle varie manopole in un forno domestico”.
Nello specifico, dato un forno, il nostro algoritmo, attraverso una rete neurale addestrata ad hoc, deve essere in grado di riconoscere se non si sta toccando alcuna manopola oppure, se si, quale nello specifico.

In questo progetto si è dovuto lavorare su delle immagini, utilizzate come input per la nostra rete. Motivo per cui è stata utilizzata una Rete Neurale Convoluzionale (CNN) che permette di applicare le reti neurali al processamento di immagini, riuscendo a scalare immagini di grandi dimensioni e grossi dataset di immagini.

Questo problema della classificazione del tocco/non tocco in un forno domestico, in realtà, può essere esteso a macchinari industriali pensando a delle azioni che conseguono da determinate scelte. Ad esempio, è possibile pensare che dopo aver toccato un determinato pulsante, il sistema dia avvio a una specifica azione o mostri delle istruzioni inerenti a quel pulsante.


PS: se si vuole replicare l'allenamento della rete bisogna creare nuovamente il dataset dividendo i video manualmente o con qualsiasi altro software i all'interno della cartella Video test e dividerle  nelle 4 cartelle di dataset (left,right,center e null)

#Usage
Per usare la seguente rete neurale bisogna lanciare lo script Progetto.py conn il path del video da analizzare
