# JetAuto Autonomous — Guida laterale + waypoint navigation

Pacchetto standalone di guida autonoma per JetAuto (Mecanum + Jetson Nano)
basato su segmentazione semantica (SegFormer(Abbandonato perche troppo pesante)/MobileNetV3 (Attuale)/TensorRT esterno). )

**NON usa catkin_ws.** Tutti gli script si lanciano direttamente con
`python2` (interprete di sistema dove ROS Melodic è installato).

## Architettura

```
                    roscore + driver Hiwonder
                    (avviati allo startup)
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
  /lane_mask         /jetauto_controller   /odom
  (publisher)        /cmd_vel              (publisher)
        ↑                  ↑
        │                  │ Twist
   ┌────────┐    ┌────────────────────┐
   │ lane_  │    │ lane_controller    │
   │ follo- │--->│ + waypoint_manager │
   │ wer.py │    │ + serve_dashboard  │
   │ (suo)  │    │ (nostri)           │
   └────────┘    └────────────────────┘
   Python 3.6         Python 2.7
   conda env           sistema
```

I due "blocchi" Python sono completamente separati a livello di processo
e ambiente; comunicano solo via topic ROS.

## File del pacchetto

| File | Ruolo |
|---|---|
| `scripts/lane_controller_node.py` | Nodo ROS che fa il controllo laterale (3 stati: TRACKING / SINGLE / STOP) |
| `scripts/waypoint_manager_node.py` | Nodo ROS che esegue Dijkstra + sequenza waypoint + gestione incroci |
| `scripts/map_loader.py` | Caricamento YAML + grafo NetworkX |
| `scripts/serve_dashboard.py` | Mini server HTTP standalone per la dashboard |
| `config/lane_params.yaml` | Tutti i parametri (gain P, soglie, BEV, ecc.) |
| `web/dashboard.html` | UI: feed video + mappa SVG + controlli |
| `maps/map_clean-edited_smooth.yaml` | La mappa della pista |
| `start_all.sh` | Avvia tutto il sistema |
| `stop_all.sh` | Ferma tutto |

## Prerequisiti sul Jetson

Lo confermi con questi comandi:

```bash
# 1) roscore deve essere attivo (lo è già all'avvio del Jetson)
pgrep -af rosmaster

# 2) Python 2 con tutte le dipendenze
python2 -c "import rospy, cv_bridge, yaml, networkx; print('OK')"

# 3) rosbridge_server e web_video_server installati
rospack find rosbridge_server
rospack find web_video_server
```

Se manca `yaml` o `networkx` per Python 2:

```bash
sudo apt install python-yaml python-networkx
```

## Installazione

Il pacchetto va in una directory qualunque (consigliato: `~/jetauto_autonomous`).

Da PC sviluppo (Mac), via scp:

```bash
cd ~/path/to/UniDrive
scp -r ros_autonomous jetauto@<IP_JETSON>:~/jetauto_autonomous
```

Sul Jetson:

```bash
cd ~/jetauto_autonomous
chmod +x start_all.sh stop_all.sh scripts/*.py
```

## Avvio

In un terminale, sul Jetson:

```bash
cd ~/jetauto_autonomous
./start_all.sh
```

Cosa lancia:
1. carica `config/lane_params.yaml` in `rosparam`
2. `rosbridge_websocket` (porta 9090) — comunicazione WebSocket per la dashboard
3. `web_video_server` (porta 8080) — streaming MJPEG dei topic immagine
4. `serve_dashboard.py` (porta 8000) — server HTTP della dashboard
5. `lane_controller_node.py` — controllo laterale
6. `waypoint_manager_node.py` — gestione waypoint

I log finiscono in `/tmp/jetauto_autonomous_logs/`.
I PID dei processi in `/tmp/jetauto_autonomous.pids`.

## In un altro terminale: avvia il SegFormer

In **un terminale separato** (perché vive in un altro ambiente Python):

```bash
conda activate <env_name>
cd ~/path/to/UniDrive/on_jetauto_scripts/drive_segm
python lane_follower.py [args che usa di solito]
```


Verifica con:

```bash
rostopic hz /lane_mask
```

## Apri la dashboard

Da qualunque PC sulla stessa rete (o dallo stesso Jetson via NoMachine):

```
http://<IP_JETSON>:8000/
```

Devi vedere:
- pallino "Connected" verde in alto a destra
- feed video con maschera colorata
- mappa SVG con i 156 nodi della pista
- controlli Start/End, START, STOP

## Stop

```bash
cd ~/jetauto_autonomous
./stop_all.sh
```

## Modalità test (senza modello acceso)

Per validare la pipeline di controllo + dashboard senza dover lanciare
SegFormer/SegNet (utile per debugging, sviluppo, e per misurare l'occupazione
RAM dello stack senza il modello), c'è uno script alternativo
`start_test.sh` che lancia un finto publisher di `/lane_mask`.

### 1) Genera le maschere di test dai JSON LabelMe del dataset

Sul Jetson:

```bash
cd ~/jetauto_autonomous

# Cartella sorgente con i JSON (e relative immagini)
DATASET_DIR=/path/al/dataset/originale

# Genera le maschere a 512x256 (alleggerisce il carico)
python2 scripts/labelme_to_mask.py "$DATASET_DIR" \
    -o ./test_masks --resize 512x256 --debug
```

`--debug` salva anche `*_vis.png` colorati per controllo visivo.
Verifica una maschera a campione aprendo un file `_vis.png`.

### 2) Avvia il sistema in modalità test

```bash
# Sequenziale, 10 Hz (simula un video del modello)
./start_test.sh

# Random ordering, ogni maschera tenuta 0.5 secondi
./start_test.sh --random --hold 0.5

# Frequenza più alta (stress test del controller)
./start_test.sh --rate 20
```

A questo punto la dashboard funziona come al solito ma le maschere
arrivano dal publisher fake invece che dal modello.

**ATTENZIONE**: il `/jetauto_controller/cmd_vel` viene comunque pubblicato.
Tenere il robot **sollevato** o disattiva i motori se non vuoi che si muova.

### 3) Controllo a runtime del fake publisher

```bash
# Pausa la sequenza (utile per inspezionare un singolo frame)
rostopic pub /fake_mask_publisher/cmd std_msgs/String "data: 'pause'"

# Riprende
rostopic pub /fake_mask_publisher/cmd std_msgs/String "data: 'resume'"

# Avanza al prossimo frame manualmente (in pause)
rostopic pub /fake_mask_publisher/cmd std_msgs/String "data: 'next'"
```

Manda `kill` a tutti i processi salvati nel PID file e pubblica anche un
`Twist` zero per fermare il robot a velocità nulla.

## Override veloci

```bash
./start_all.sh --bev      # forza input_mode=bev_topic (usa /lane_mask già warpato)
./start_all.sh --camera   # forza input_mode=camera (default)
```

## Tuning rapido

Modifica `config/lane_params.yaml` direttamente sul Jetson (è puro YAML, non
serve ricompilare niente). Riavvia con:

```bash
./stop_all.sh && ./start_all.sh
```

Parametri che probabilmente vorrai toccare al primo test:

| Parametro | Effetto | Default | Range tipico |
|---|---|---|---|
| `linear_x_speed` | velocità di crociera (m/s) | 0.10 | 0.05 - 0.25 |
| `Kp_lat` | gain laterale (px → m/s) | 0.0030 | 0.001 - 0.008 |
| `lateral_offset_px` | distanza dalla linea singola | 60 | 30 - 120 |
| `mask_roi_top_fraction` | quanta parte della maschera analizzare | 0.55 | 0.4 - 0.7 |
| `no_lane_grace_frames` | frame consecutivi senza linee prima di STOP | 3 | 2 - 8 |
| `control_rate_hz` | frequenza loop controllo | 20 | regola sui FPS del modello |

Misura gli FPS del modello con:

```bash
rostopic hz /lane_mask
```

E imposta `control_rate_hz` ≤ FPS_modello + 5.

## Troubleshooting

**"roscore/rosmaster NON è attivo"** → il sistema non è ancora pronto. Aspetta
che lo startup-script finisca, oppure lancialo a mano: `roscore &`.

**"dipendenze Python mancanti"** → `sudo apt install python-yaml python-networkx`.

**Robot non si muove dopo START** → controlla:
- `rostopic hz /lane_mask` deve dare frequenza > 0
- `rostopic echo /lane_controller/state` deve mostrare `TRACKING_*`, non `STOP`
- `rostopic echo /jetauto_controller/cmd_vel` deve mostrare Twist non-zero

**Robot oscilla** → abbassa `Kp_lat` a 0.002 o aumenta `smooth_alpha` a 0.6.

**Dashboard "Disconnected"** → verifica che la porta 9090 sia raggiungibile
dal browser: `curl http://<IP>:9090/` deve dare almeno una risposta HTTP.

**Feed video grigio** → `tail -f /tmp/jetauto_autonomous_logs/web_video_server.log`
per capire se ci sono errori. Verifica `rostopic hz /lane_debug/image`.
