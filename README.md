# Rilevamento dell'utilizzo della mascherina con supporto della stima della posa

## Requisiti

Requisiti di base:
```
numpy == 1.17
opencv-python >= 4.1
torch >= 1.6
torchvision
matplotlib
pycocotools
tqdm
pillow
tensorboard >= 1.14
```
Libreria per utlizzare la funzione di attvazione Mish con supporto CUDA in Pytorch:
```
git clone https://github.com/thomasbrandon/mish-cuda
cd mish-cuda
python setup.py build install
```
Nel caso non funzionasse utlizzare questa come alternativa:
```
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python setup.py build install
```
Per parsare i file YAML in python:
```
pip install -U PyYAML
```
Per utilizzare la pose estimation all'interno del rilevamento è preferibile avere anche il modulo DNN di OpenCV.

Per la sua installazione è necessaria una GPU CUDA compatibile con istallati sopra il CUDA Toolkit (preferibile la versione 10.1) e la libreria cuDNN (preferibile la versione 7.5.0)

Per eseguire l'istallazione in ambiente linux (ad esempio quando si utilizza un sistema di cloud GPU come Google Colab)
```
%cd /content
!git clone https://github.com/opencv/opencv
!git clone https://github.com/opencv/opencv_contrib
!mkdir /content/build
%cd /content/build

!cmake -DOPENCV_EXTRA_MODULES_PATH=/content/opencv_contrib/modules  \
       -DBUILD_SHARED_LIBS=OFF \
       -DBUILD_TESTS=OFF \
       -DBUILD_PERF_TESTS=OFF \
       -DBUILD_EXAMPLES=OFF \
       -DWITH_OPENEXR=OFF \
       -DWITH_CUDA=ON \
       -DWITH_CUBLAS=ON \
       -DWITH_CUDNN=ON \
       -DOPENCV_DNN_CUDA=ON \
       /content/opencv

!make -j8 install

import cv2
cv2.__version__
```
Su windows invece eseguire l'installazione con l'ausilio di CMAKE e VisualStudio

## Struttura delle cartelle
- **data** contiene i file YAML per la configurazione di dataset e iperparametri
- **models** contiene le classi e i file di configurazione necessari per utilizzare il modello
- **runs** contiene le esecuzione delle sessioni di addestramento e di rilevamento
- **inference** contiene le/i immagini/video su cui si vuole esegure la rilevazione
- **weights** contiene pesi pre-addestrati utili per inizializzare l'addestramento
- **utils** contiene tutte le varie funzioni di utilità necessarie a far funzionale il framework
## Addestramento
Per addestrare la rete è necessario far partire il file train.py da linea di comando passadogli tutti gli argomenti necessari.

La lista di alcuni degli argomenti più utili o necessari è la seguente:

- --weights: La stringa rappresentante il path dei pesi con cui inizializzare il modello
- --cfg: Stringa rappresentante il path del file YAML contenente la configurazione dell'architettura che si vuole utilizzare
- --data: Stringa rappresentante il path del file YAML contenente la configurazione del dataset
- --epochs: L'intero rappresentante il numero di epoche necessarie per addestrare la rete
- --batch-size: L'intero rappresentante la dimensione del batch
- --img-size: La dimensione verso cui si volgliono ridimensionare le immagini in input
- --resume: Flag utilizzata per riprendere l'addestramento 
- --cache-images: Flag utilizzata per creare una cache della immagini analizzate all'interno del dataset, rendendo così più veloce il prossimo addestramento
- --name: Nome della cartella dentro cui si vogliono salvare i dati di addestramento
- --adam: Flag da utilizzare nel caso si volesse usare Adam come Optimizer
- --hyp: Stringa rappresentante il path del file YAML contenente la configurazione degli iperparametri

Esempio di un addestramento:
```
python train.py --batch 16 -- epoch 300 --data mask.yaml --cfg ./models/yolov4-csp.yaml --weights '' --name nome_cartella_addestramento
```
Esempio di ripresa dell'addestramento dall'ultimo checkpoint:
```
python train.py --batch 16 -- epoch 300 --data mask.yaml --cfg ./models/yolov4-csp.yaml --weights 'runs/exp0_nome_cartella_addestramento/weights/last.pt' --name nome_cartella_addestramento2
```

## Rilevamento
Per inferire la rete addestrata è necessario far partire il file detect.py da linea di comando passandogli tutti gli argomenti necessari.

La lista di alcuni degli argomenti più utili o necessari è la seguente:

- --weights: La stringa rappresentante il path del modello addestrato che si intende utilizzare per il rilevamento
- --img-size: La dimensione verso cui si volgliono ridimensionare le immagini in input 
- --name: Nome della cartella dentro cui si vogliono salvare i risultati del rilevamento
- --source: La stringa rappresentante il path della cartella contenente le immagini e i video da analizzare
- --conf-thres: La soglia di affidabilità minima per prendere in considerazione una predizione
- --iou-thres: Il valore minimo Intersection over Union che deve avere un bounding box per essere preso in considerazione 
- --exist-ok: Se la cartella esiste già non crearne un'altra ma sovrescivere quella vecchia
- --openpose: Flag utilizzata per caricare in memoria il modello di pose estimation
- --op-thres: Il valore utilizzato per decrementare l'affidabilità in caso di utilizzo della pose estimation

Esempio di un rilevamento senza pose estimation:
```
python detect.py --weights ./runs/train/cartella_di_salvataggio_del_modello/weights/best.pt --conf 0.6 --source ./inference/images --name cartella_di_salvataggio --exist-ok
```

Esempio di un rilevamento con pose estimation:
```
python detect.py --weights ./runs/train/cartella_di_salvataggio_del_modello/weights/best.pt --conf 0.6 --source ./inference/images --name cartella_di_salvataggio --exist-ok --openpose --op-thres 0.2
```