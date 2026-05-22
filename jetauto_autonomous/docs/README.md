# JetAuto Autonomous - Guida laterale + waypoint navigation

Pacchetto standalone di guida autonoma per JetAuto (Mecanum + Jetson Nano)
basato su segmentazione semantica (MobileNetV3 / TensorRT esterno).

**NON usa catkin_ws.** Tutti gli script si lanciano direttamente con
`python2` (interprete di sistema dove ROS Melodic è installato).

## Piattaforma

### Hardware

| Voce | Valore |
|---|---|
| Board | NVIDIA Jetson Nano Developer Kit |
| Module | NVIDIA Jetson Nano (16 GB eMMC) |
| SoC | Tegra210 (Porg) |
| CUDA Arch | 5.3 |
| L4T / JetPack | 32.7.4 / 4.6.4 |
| Hostname | `jetauto` |

### Software

| Voce | Valore |
|---|---|
| OS | Ubuntu 18.04 Bionic Beaver |
| Kernel | 4.9.337-tegra |
| Python (sistema / ROS) | 2.7 |
| Python (lane follower, conda) | 3.6.9 |
| ROS | Melodic |
| CUDA | 10.2.300 |
| cuDNN | 8.2.1.32 |
| TensorRT | 8.2.1.8 |
| OpenCV | 4.5.5 (CUDA: YES) |
| VPI / Vulkan | 1.2.3 / 1.2.70 |

## Architettura

```
                roscore + driver Hiwonder
                (avviati allo startup)
                       │
    ┌------------------┼------------------┐
    ↓                  ↓                  ↓
/lane_mask_bev    /jetauto_controller   /odom
(publisher)       /cmd_vel              (publisher)
    ↑                  ↑
    │                  │ Twist
┌--------┐    ┌------------------------------------┐
│ lane_  │    │ lane_controller_node               │
│ follo- │--->│  └- lane_core (pura logica)        │
│ wer.py │    │ + waypoint_manager_node            │
│ (suo)  │    │ + map_follower_node (fallback BEV) │
│        │    │ + serve_dashboard                  │
└--------┘    └------------------------------------┘
Python 3.6.9       Python 2.7
 conda env          sistema
```

I due "blocchi" Python sono completamente separati a livello di processo
e ambiente; comunicano solo via topic ROS.

## File del pacchetto

| File | Ruolo |
|---|---|
| `scripts/lane_controller_node.py` | Nodo ROS: wiring rosparam/pub/sub, gestisce stati TRACKING/SINGLE/STOP/DISABLED |
| `scripts/lane_core.py` | Logica pura (no ROS): Hough, fit polinomiale, PD sterzata - importabile anche dall'offline tester |
| `scripts/waypoint_manager_node.py` | Nodo ROS: Dijkstra + sequenza waypoint + controllo diretto agli incroci |
| `scripts/map_follower_node.py` | Fallback pure-pursuit: guida sul grafo mappa quando le corsie spariscono |
| `scripts/map_loader.py` | Caricamento YAML + grafo NetworkX, classificazione nodi |
| `scripts/serve_dashboard.py` | Mini server HTTP standalone per la dashboard |
| `config/lane_params.yaml` | Tutti i parametri (gain P, soglie, BEV, map_follower, ecc.) |
| `web/dashboard.html` | UI: feed video + mappa SVG + controlli |
| `maps/map_clean-edited_smooth.yaml` | La mappa della pista |
| `start_all.sh` | Avvia tutto il sistema (8 processi) |
| `stop_all.sh` | Ferma tutto + pubblica zero Twist |

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

Da PC sviluppo, via scp:

```bash
cd ~/path/to/UniDrive
scp -r jetauto_autonomous jetauto@<IP_JETSON>:~/jetauto_autonomous
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
1. Carica `config/lane_params.yaml` in `rosparam`
2. `rosbridge_websocket` (porta 9090) - comunicazione WebSocket per la dashboard
3. `web_video_server` (porta 8080) - streaming MJPEG dei topic immagine
4. `serve_dashboard.py` (porta 8000) - server HTTP della dashboard
5. `lane_controller_node.py` - controllo laterale
6. `waypoint_manager_node.py` - gestione waypoint
7. `map_follower_node.py` - fallback BEV (avviato sempre, attivo solo se `map_follower.enable: true`)

I log finiscono in `/tmp/jetauto_autonomous_logs/`.
I PID dei processi in `/tmp/jetauto_autonomous.pids`.

## In un altro terminale: avvia il lane follower

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

## Override veloci

```bash
./start_all.sh --bev      # forza input_mode=bev_topic (usa /lane_mask_bev già warpato)
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
| `linear_x_speed` | velocità di crociera (m/s) | 0.05 | 0.03 - 0.15 |
| `max_angular_z` | sterzata massima (rad/s, drive=classic) | 0.80 | 0.4 - 1.2 |
| `max_steering_angle` | angolo max mappato (gradi) | 48.0 | 30 - 60 |
| `single_line_offset` | offset px stima centro con singola linea | 0 | 0 - 60 |
| `hough_roi_top_frac` | porzione superiore BEV ignorata (use_bev=true) | 0.30 | 0.0 - 0.6 |
| `no_bev_roi_top_frac` | porzione superiore ignorata (use_bev=false) | 0.45 | 0.3 - 0.6 |
| `hough_threshold` | voti minimi HoughLinesP | 50 | 30 - 80 |
| `hough_max_gap_px` | gap max per unire segmenti Hough | 40 | 10 - 60 |
| `hough_min_length_px` | lunghezza min linea validata | 20 | 10 - 40 |
| `center_y_ratio` | quota di misura del centro corsia [0=top,1=bot] | 0.50 | 0.3 - 0.7 |
| `angle_smooth_alpha_base` | base EMA sull'angolo (più basso = più smooth) | 0.50 | 0.3 - 0.8 |
| `lane_width_px` | larghezza corsia in BEV (fallback statico) | 280 | 200 - 350 |
| `control_rate_hz` | frequenza loop controllo (Hz) | 20 | regola sui FPS del modello |

Misura gli FPS del modello con:

```bash
rostopic hz /lane_mask
```

E imposta `control_rate_hz` ≤ FPS_modello + 5 (vedi sezione "Tuning consigliato per Jetson Nano 4GB" più sotto).

## Calibrazione dinamica della larghezza corsia

Il controller misura continuamente la distanza fra linea sinistra e destra
quando entrambe sono visibili e mantiene una stima EMA della larghezza
corsia in pixel BEV. Quando poi il robot vede una sola linea (es. è
sbilanciato lateralmente e l'altra esce dal frame), usa la stima
dinamica al posto di `lane_width_px` statico per ricostruire il centro
corsia. Senza questo meccanismo, una `lane_width_px` errata di anche
solo il 15% rispetto alla pista reale spinge il robot sistematicamente
fuori centro.

**Quando si attiva**: solo in modalità BEV (`use_bev: true`). Il primo
frame con due linee valide fa il bootstrap. Ogni nuova misura entra
nell'EMA se passa due sanity-check:

1. Range assoluto `[lane_width_min_px, lane_width_max_px]` (sempre).
2. Banda relativa `lane_width_sanity_band` rispetto al valore corrente (solo dopo il bootstrap).

**Disabilitazione**: `lane_width_dynamic_enable: false` -> torna al
comportamento statico (usa sempre `lane_width_px`).

**Verifica dal debug image** (`/lane_debug/image`):
- In basso a sinistra appare `W=...` (giallo): valore EMA corrente in px.
- Sotto: `Wm=...` ultima misura grezza, **verde** se accettata nell'EMA, **rosso** se scartata.
- Sulla riga magenta (quota `center_y`), due tick arancioni a `±W/2` dal centro corsia stimato.

**Tuning**:
- Se `W` oscilla di ±20px frame su frame -> abbassa `lane_width_ema_alpha` (es. 0.05).
- Se `Wm` è spesso rosso anche su pista buona -> allarga `lane_width_sanity_band` (es. 0.35) o ricontrolla i bound assoluti.
- Se `W=--` permanente -> nessuna misura ha mai passato i sanity-check; controlla `lane_width_min_px` / `lane_width_max_px` (sono in pixel **post-`bev_scale`**).

**Scaling con bev_scale**: se imposti `bev_scale: 2.0`, raddoppia
`lane_width_px`, `lane_width_min_px`, `lane_width_max_px`. Il nodo
emette un warning a startup se i bound non comprendono `lane_width_px`.

## map_follower_node - fallback BEV

`map_follower_node.py` è un controller di fallback che guida il robot sul
grafo della mappa quando le corsie non sono visibili (es. incroci privi di
segnaletica, zone danneggiate della pista).

**Quando si attiva** (tutte le condizioni devono essere vere):
- `lane_controller/state == HOLD` per `hold_fallback_frames` tick consecutivi
- `waypoint_manager/status == NAVIGATING`
- NOT in stato JUNCTION

**Cosa fa**: pure-pursuit sul path calcolato da `waypoint_manager_node`, usando
l'odometria. Mentre è attivo pubblica `enable=False` su `/lane_controller/enable`
(arbitration), e lo riabilita con `enable=True` una volta che la corsia è tornata
stabile.

**Abilitazione**: il nodo è sempre avviato da `start_all.sh` ma disabilitato di
default. Per attivarlo:

```yaml
# config/lane_params.yaml
map_follower:
  enable: true
```

oppure a caldo (senza riavviare):

```bash
rosparam set /map_follower/enable true
```

**Parametri chiave**:

| Parametro | Default | Effetto |
|---|---|---|
| `hold_fallback_frames` | 15 (= 1.5 s @ 10 Hz) | Tick in HOLD prima dell'attivazione |
| `lane_recovery_frames` | 5 (= 0.5 s) | Tick di corsia stabile prima del ritorno |
| `lookahead_m` | 0.50 m | Distanza pure-pursuit |
| `map_drive_speed` | 0.04 m/s | Velocità in modalità mappa |
| `angular_kp` | 1.2 | Guadagno P errore heading - angular.z |

**Topic di debug**:
- `/map_follower/state` - stato corrente: INACTIVE / ACTIVATING / ACTIVE / RECOVERING
- `/map_follower/active` - Bool, latched

## Tuning consigliato per Jetson Nano 4GB

Con MobileNetV3 + TensorRT esterno il modello produce 20-30 FPS sul
Jetson Nano 4GB. Valori suggeriti per `lane_params.yaml`:

```yaml
control_rate_hz: 15        # margine su latenza ROS Melodic (Python 2)
bev_scale: 1.0             # alzare a 2.0 solo se il Jetson regge ed è davvero utile
hough_threshold: 50
hough_min_line_px: 20
hough_max_gap_px: 20       # su BEV 320×128 un gap di 40 unisce segmenti distanti
hough_min_length_px: 20
hough_roi_top_frac: 0.30
center_y_ratio: 0.50
lane_fit_mode: "auto"
angle_smooth_alpha_base: 0.50
lane_width_ema_alpha: 0.10
```

Note:
- `control_rate_hz` deve essere ≤ `FPS_modello`. Con 25 FPS reali, 15 Hz lascia margine alla latenza ROS Melodic su Python 2.
- `hough_max_gap_px=40` (default storico) su BEV 320×128 può unire segmenti che appartengono a linee diverse: con linee tratteggiate e rumore, 20 è più conservativo.
- Se la CPU del Jetson è satura, abbassa `publish_debug: false` o `debug_scale: 0.4`.

## Troubleshooting

**"roscore/rosmaster NON è attivo"** -> il sistema non è ancora pronto. Aspetta
che lo startup-script finisca, oppure lancialo a mano: `roscore &`.

**"dipendenze Python mancanti"** -> `sudo apt install python-yaml python-networkx`.

**Robot non si muove dopo START** -> controlla:
- `rostopic hz /lane_mask` deve dare frequenza > 0
- `rostopic echo /lane_controller/state` deve mostrare `TRACKING_*`, non `STOP`
- `rostopic echo /jetauto_controller/cmd_vel` deve mostrare Twist non-zero

**Robot si ferma in mezzo alla pista senza corsia visibile** - con
`map_follower.enable: false` il sistema entra in HOLD e si ferma.
Per attivare il fallback:

```bash
rosparam set /map_follower/enable true
```

oppure editare `config/lane_params.yaml` e riavviare.

**Robot oscilla** -> abbassa `Kp_lat` a 0.002 o aumenta `smooth_alpha` a 0.6.

**Dashboard "Disconnected"** -> verifica che la porta 9090 sia raggiungibile
dal browser: `curl http://<IP>:9090/` deve dare almeno una risposta HTTP.

**Feed video grigio** -> `tail -f /tmp/jetauto_autonomous_logs/web_video_server.log`
per capire se ci sono errori. Verifica `rostopic hz /lane_debug/image`.
