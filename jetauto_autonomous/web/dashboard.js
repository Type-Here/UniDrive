// CONFIG and NAV_CONFIRM_DIST are defined inline in dashboard.html
// (they contain __VIDEO_SERVER_IP__ substituted by the server at startup)

// --- ROS ----------------------------------------------------------------------
const ros    = new ROSLIB.Ros({ url: CONFIG.rosbridge_url });
const connEl = document.getElementById('connStatus');

ros.on('connection', () => {
  connEl.classList.add('ok');
  connEl.children[1].textContent = 'Connected';
  setupTopics();
  switchTopic(currentTopicMode);
});
ros.on('error', () => { connEl.classList.remove('ok'); connEl.children[1].textContent = 'Error'; });
ros.on('close', () => { connEl.classList.remove('ok'); connEl.children[1].textContent = 'Disconnected'; });

// --- Video feed ---------------------------------------------------------------
let currentTopicMode = 'camera';

function topicForMode(mode) {
  if (mode === 'camera')  return CONFIG.topic_camera;
  if (mode === 'lfDebug') return CONFIG.topic_lfDebug;
  if (mode === 'objDet')  return CONFIG.topic_objDet;
  return CONFIG.topic_debug;
}

function labelForMode(mode) {
  if (mode === 'camera')  return CONFIG.topic_camera;
  if (mode === 'lfDebug') return CONFIG.topic_lfDebug;
  if (mode === 'objDet')  return CONFIG.topic_objDet;
  return CONFIG.topic_debug;
}

function switchTopic(mode) {
  currentTopicMode = mode;
  ['camera', 'debug', 'lfDebug', 'objDet'].forEach(m => {
    const id = 'btn' + m.charAt(0).toUpperCase() + m.slice(1);
    document.getElementById(id).classList.toggle('active', m === mode);
  });
  document.getElementById('videoSourceLabel').textContent = labelForMode(mode);
  renderVideo(topicForMode(mode));
}

function renderVideo(topic) {
  const wrap  = document.getElementById('videoWrap');
  const label = document.getElementById('videoSourceLabel');
  // Differentiated quality by resolution: camera 640x420 -> 30, debug 320x128 -> 65, bev 640x128 -> 40
  const QUALITY = {};
  QUALITY[CONFIG.topic_camera]  = 30;
  QUALITY[CONFIG.topic_debug]   = 65;
  QUALITY[CONFIG.topic_lfDebug] = 40;
  QUALITY[CONFIG.topic_objDet]  = 50;
  const quality = QUALITY[topic] !== undefined ? QUALITY[topic] : 40;
  const url   = `${CONFIG.video_server_url}/stream?topic=${topic}&type=mjpeg&quality=${quality}`;
  wrap.innerHTML = '';
  wrap.appendChild(label);                 // re-attach the overlay label
  const img = document.createElement('img');
  img.alt = 'video feed';
  img.onerror = () => {
    wrap.innerHTML = '';
    wrap.appendChild(label);
    const ph = document.createElement('div');
    ph.className = 'ph';
    ph.textContent = `Feed non disponibile: ${topic}`;
    wrap.appendChild(ph);
  };
  img.src = url;
  wrap.appendChild(img);
}

// --- Drive enable / disable ---------------------------------------------------
let laneEnablePub     = null;
let remapTransformPub = null;   // std_msgs/Float64MultiArray [theta, scale, tx, ty]
let remapCmdPub       = null;   // pre-advertised cmd_vel for Remap (avoids advertise latency)
let driveEnabled      = false;

function setDriveEnabled(val) {
  if (!laneEnablePub) { alert('ROS non connesso'); return; }
  laneEnablePub.publish(new ROSLIB.Message({ data: val }));
  driveEnabled = val;
  syncDriveUI(val);
}

function syncDriveUI(val) {
  const el = document.getElementById('driveState');
  if (val) { el.textContent = '● IN MARCIA'; el.className = 'drive-state running'; }
  else     { el.textContent = '● FERMA';      el.className = 'drive-state stopped'; }
}

// --- ROS topics ---------------------------------------------------------------
let goalPub = null;

function setupTopics() {
  goalPub = new ROSLIB.Topic({
    ros, name: '/waypoint_manager/goal', messageType: 'std_msgs/Int32MultiArray'
  });
  laneEnablePub = new ROSLIB.Topic({
    ros, name: '/lane_controller/enable', messageType: 'std_msgs/Bool'
  });
  remapTransformPub = new ROSLIB.Topic({
    ros, name: '/remap_transform', messageType: 'std_msgs/Float64MultiArray'
  });
  remapCmdPub = new ROSLIB.Topic({
    ros, name: '/jetauto_controller/cmd_vel', messageType: 'geometry_msgs/Twist'
  });

  new ROSLIB.Topic({ ros, name: '/waypoint_manager/status', messageType: 'std_msgs/String', throttle_rate: 500, queue_length: 1 })
    .subscribe(m => updateNavBadge(m.data));

  new ROSLIB.Topic({ ros, name: '/waypoint_manager/path', messageType: 'std_msgs/Int32MultiArray', throttle_rate: 500, queue_length: 1 })
    .subscribe(m => highlightPath(m.data));

  new ROSLIB.Topic({ ros, name: '/lane_controller/state', messageType: 'std_msgs/String', throttle_rate: 500, queue_length: 1 })
    .subscribe(m => {
      updateLaneBadge(m.data);
      // Sync the drive button with the actual controller state
      if (m.data === 'DISABLED') {
        driveEnabled = false; syncDriveUI(false);
      } else if (!driveEnabled && m.data !== 'STOP') {
        driveEnabled = true; syncDriveUI(true);
      }
    });

  new ROSLIB.Topic({ ros, name: '/odom', messageType: 'nav_msgs/Odometry', throttle_rate: 200, queue_length: 1 })
    .subscribe(m => updateOdom(m));

  new ROSLIB.Topic({ ros, name: '/orchestrator/state', messageType: 'std_msgs/String', throttle_rate: 500, queue_length: 1 })
    .subscribe(m => {
      const b = document.getElementById('mapFollowerBadge');
      const s = m.data;
      if (s === 'JUNCTION') {
        b.style.display = '';
        b.className = 'badge junction';
        b.textContent = 'JUNCTION';
      } else if (s === 'FALLBACK') {
        b.style.display = '';
        b.className = 'badge map-follower';
        b.textContent = 'MAP FALLBACK';
      } else {
        b.style.display = 'none';
      }
    });

  new ROSLIB.Topic({ ros, name: '/waypoint_manager/nav_info', messageType: 'std_msgs/Float64MultiArray', throttle_rate: 200, queue_length: 1 })
    .subscribe(m => {
      if (!svgViewBox || !mapData || m.data.length < 8) return;
      const isActive = m.data[7] > 0.5;
      const nodeId   = isActive ? Math.round(m.data[0]) : -1;
      const tm = document.getElementById('targetMarker');
      if (!tm) return;
      if (nodeId >= 0 && mapData.nodes[nodeId]) {
        const node = mapData.nodes[nodeId];
        tm.setAttribute('cx', svgViewBox.tx(node.x));
        tm.setAttribute('cy', svgViewBox.ty(node.y));
        tm.style.display = '';
      } else {
        tm.style.display = 'none';
      }
    });
}

// --- Lane / Nav badges --------------------------------------------------------
const LANE_CLASS = {
  TRACKING_CC:'tracking-cc', TRACKING_DC:'tracking-dc',
  SINGLE_LINE:'single-line', SINGLE_DASHED:'single-dashed',
  GRACE:'grace', STOP:'stop', DISABLED:'disabled', NONE:'stop',
};
const LANE_LABEL = {
  TRACKING_CC:'TRACKING (2 cont.)', TRACKING_DC:'TRACKING (dashed+cont.)',
  SINGLE_LINE:'SINGLE LINE', SINGLE_DASHED:'SINGLE DASHED',
  GRACE:'GRACE (recovering)', STOP:'STOP', DISABLED:'DISABLED',
};

function updateLaneBadge(s) {
  const b = document.getElementById('laneState');
  b.textContent = 'LANE: ' + (LANE_LABEL[s] || s);
  b.className   = 'badge ' + (LANE_CLASS[s] || '');
}

function updateNavBadge(s) {
  const b = document.getElementById('navState');
  b.textContent = 'NAV: ' + s;
  if      (s.startsWith('NAVIGATING'))  b.className = 'badge nav';
  else if (s.startsWith('GOAL'))        b.className = 'badge done';
  else if (s.startsWith('ERROR'))       b.className = 'badge stop';
  else                                   b.className = 'badge';
}

// --- Odometry -----------------------------------------------------------------
function nearestMapNode(mx, my) {
  if (!mapData) return null;
  let best = null, bestD = Infinity;
  for (const [id, n] of Object.entries(mapData.nodes)) {
    const d = Math.hypot(n.x - mx, n.y - my);
    if (d < bestD) { bestD = d; best = { id: +id, dist: d }; }
  }
  return best;
}

function updateOdom(msg) {
  const p = msg.pose.pose.position;
  const q = msg.pose.pose.orientation;
  const v = msg.twist.twist.linear;
  const siny = 2*(q.w*q.z + q.x*q.y);
  const cosy = 1 - 2*(q.y*q.y + q.z*q.z);

  // save raw position for calibration
  lastRawOdom.x = p.x;
  lastRawOdom.y = p.y;

  // transform raw odom -> map frame using current calibration/remap transform
  const mapped = odomToMap(p.x, p.y);

  document.getElementById('odomX').textContent   = mapped.x.toFixed(3);
  document.getElementById('odomY').textContent   = mapped.y.toFixed(3);
  document.getElementById('odomYaw').textContent = (Math.atan2(siny,cosy)*180/Math.PI).toFixed(1);
  document.getElementById('odomVx').textContent  = Math.hypot(v.x, v.y).toFixed(3);
  updateRobot(mapped.x, mapped.y);

  // Show nearest node estimate — helps user choose correct start before NAVIGA
  const nn = nearestMapNode(mapped.x, mapped.y);
  if (nn) {
    document.getElementById('nearestNodeId').textContent   = nn.id;
    document.getElementById('nearestNodeDist').textContent = nn.dist.toFixed(2) + ' m';
  }
}

// --- Map ----------------------------------------------------------------------
let mapData = null, svgViewBox = null;

async function loadMap() {
  // Always load the canonical YAML map (source of truth)
  try {
    const txt = await (await fetch(CONFIG.map_yaml_url)).text();
    mapData = parseMapYaml(txt);
    renderMap();
  } catch(e) {
    document.getElementById('mapSvg').innerHTML =
      `<text x="20" y="40" fill="#f87272">Errore caricamento mappa: ${e}</text>`;
    return;
  }
  // Restore persisted transform from a previous Calibra/Remap session
  try {
    const r = await fetch('remap_params.json');
    if (r.ok) {
      const p = await r.json();
      if (p && typeof p.theta === 'number') {
        mapTransform = { theta: p.theta, scale: p.scale || 1, tx: p.tx || 0, ty: p.ty || 0 };
        document.getElementById('mapFrame').textContent = (mapData.frame || 'odom') + ' ✓remapped';
      }
    }
  } catch(e) { /* no remap_params.json yet */ }
}

function parseMapYaml(text) {
  const nodes={}, edges=[];
  let frame='odom', section=null, cur=null;

  const flush = () => {
    if (!cur) return;
    if (section==='nodes' && cur.id!==undefined && cur.x!==undefined)
      nodes[cur.id] = {x:cur.x, y:cur.y};
    if (section==='edges' && cur.from!==undefined)
      edges.push({from:cur.from, to:cur.to, length:cur.length||0});
    cur = null;
  };

  for (const raw of text.split('\n')) {
    const l = raw.replace(/\r/g,'');
    if (!l.trim()) continue;

    // List item (may start at column 0): must check BEFORE top-level key test
    if (/^-\s/.test(l)) {
      flush();
      cur = {};
      const m = l.match(/^-\s+(\w+)\s*:\s*(.+)/);
      if (m) cur[m[1]] = isNaN(+m[2]) ? m[2] : +m[2];
      continue;
    }

    // Continuation property "  key: value"
    if (cur !== null && /^\s+\w+\s*:/.test(l)) {
      const m = l.match(/^\s+(\w+)\s*:\s*(.+)/);
      if (m) cur[m[1]] = isNaN(+m[2]) ? m[2] : +m[2];
      continue;
    }

    // Top-level section headers
    if (/^frame_id\s*:/.test(l))  { frame = l.split(':').slice(1).join(':').trim(); continue; }
    if (/^nodes\s*:\s*$/.test(l)) { flush(); section='nodes'; continue; }
    if (/^edges\s*:\s*$/.test(l)) { flush(); section='edges'; continue; }
    // Any other top-level key = ignore
    if (/^\w/.test(l)) { flush(); section=null; continue; }
  }
  flush();
  return {frame, nodes, edges};
}

// --- Map node picking (click on input -> then click a map node) ------------------
// activeField: null | 'start' | 'end'
let activeField = null;

function setActiveField(field) {
  activeField = field;
  ['start','end'].forEach(f => {
    document.getElementById(f+'Input').classList.toggle('picking', f === field);
  });
  document.getElementById('mapSvg').classList.toggle('picking-mode', field !== null);
}

// When the user clicks an input field, it activates node picking mode
['startInput','endInput'].forEach(id => {
  const el = document.getElementById(id);
  const field = id.replace('Input','');
  el.addEventListener('focus', () => setActiveField(field));
  el.addEventListener('blur',  () => {
    // Small delay: if blur is caused by an SVG click, don't deactivate immediately
    setTimeout(() => { if(activeField === field) setActiveField(null); }, 200);
  });
});

function renderMap() {
  const svg = document.getElementById('mapSvg');
  document.getElementById('mapFrame').textContent = mapData.frame;
  const ids = Object.keys(mapData.nodes);
  if (!ids.length) return;
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  for (const id of ids) {
    const n=mapData.nodes[id];
    if(n.x<minX)minX=n.x; if(n.x>maxX)maxX=n.x;
    if(n.y<minY)minY=n.y; if(n.y>maxY)maxY=n.y;
  }
  const mg=0.3; minX-=mg; minY-=mg; maxX+=mg; maxY+=mg;
  const W=maxX-minX, H=maxY-minY;
  const SVG_W=1000, SVG_H=1000*H/W;
  const tx=x=>((x-minX)/W)*SVG_W;
  const ty=y=>SVG_H-((y-minY)/H)*SVG_H;
  svgViewBox={tx,ty,SVG_W,SVG_H};
  svg.setAttribute('viewBox',`0 0 ${SVG_W} ${SVG_H}`);
  const deg={};
  for(const e of mapData.edges){deg[e.from]=(deg[e.from]||0)+1;deg[e.to]=(deg[e.to]||0)+1;}
  // Background grid: mg=0.3 is the cell size → 14 cols × 11 rows
  let html='';
  const nCols=Math.round(W/mg), nRows=Math.round(H/mg);
  for(let i=0;i<=nCols;i++){const sx=tx(minX+i*mg); html+=`<line class="grid-line" x1="${sx}" y1="0" x2="${sx}" y2="${SVG_H}"/>`;}
  for(let j=0;j<=nRows;j++){const sy=ty(minY+j*mg); html+=`<line class="grid-line" x1="0" y1="${sy}" x2="${SVG_W}" y2="${sy}"/>`;}
  for(const e of mapData.edges){
    const a=mapData.nodes[e.from],b=mapData.nodes[e.to];
    if(!a||!b) continue;
    html+=`<line class="edge" data-from="${e.from}" data-to="${e.to}" x1="${tx(a.x)}" y1="${ty(a.y)}" x2="${tx(b.x)}" y2="${ty(b.y)}"/>`;
  }
  for(const id of ids){
    const n=mapData.nodes[id], isJ=(deg[id]||0)>2;
    html+=`<circle class="node ${isJ?'junction':''}" data-id="${id}" cx="${tx(n.x)}" cy="${ty(n.y)}" r="4"><title>Node ${id} (${n.x.toFixed(2)}, ${n.y.toFixed(2)})</title></circle>`;
    html+=`<text class="node-label" x="${tx(n.x)+5}" y="${ty(n.y)-5}">${id}</text>`;
  }
  html+=`<circle id="robotMarker" class="robot"  cx="0" cy="0" r="6" style="display:none"/>`;
  html+=`<circle id="targetMarker" class="target" cx="0" cy="0" r="4" style="display:none"/>`;
  svg.innerHTML=html;

  svg.querySelectorAll('.node').forEach(el=>{
    el.addEventListener('mousedown', ev => {
      // preventDefault prevents blur on the input from propagating before the click
      ev.preventDefault();
    });
    el.addEventListener('click', ev => {
      const id = el.getAttribute('data-id');
      if (activeField) {
        // Picking mode: assign to the active field
        document.getElementById(activeField+'Input').value = id;
        setActiveField(null);
      } else if (ev.shiftKey) {
        document.getElementById('endInput').value = id;
      } else {
        document.getElementById('startInput').value = id;
      }
      previewPath();
    });
  });
}

// --- Robot position calibration -----------------------------------------------
// mapTransform: 2D similarity transform  odom = scale * R(θ) * map + t
// inverse (odomToMap):                   map  = R(-θ) / scale * (odom - t)
let mapTransform = { theta: 0, scale: 1, tx: 0, ty: 0 };
let lastRawOdom  = { x: 0, y: 0 };   // last raw position received from /odom
let currentPath  = [];                // active path received from /waypoint_manager/path

// --- Route snapping ----------------------------------------------------------
// Project (x,y) onto the nearest segment of the active path.
// Like Google Maps: the marker follows the road, not the raw odom.
function snapToPath(x, y) {
  if (!mapData || currentPath.length < 2) return { x, y };
  let bestDist = Infinity, bestX = x, bestY = y;
  for (let i = 0; i < currentPath.length - 1; i++) {
    const a = mapData.nodes[currentPath[i]];
    const b = mapData.nodes[currentPath[i + 1]];
    if (!a || !b) continue;
    const dx = b.x - a.x, dy = b.y - a.y;
    const len2 = dx * dx + dy * dy;
    const t = len2 > 0 ? Math.max(0, Math.min(1, ((x - a.x) * dx + (y - a.y) * dy) / len2)) : 0;
    const px = a.x + t * dx, py = a.y + t * dy;
    const d = Math.hypot(x - px, y - py);
    if (d < bestDist) { bestDist = d; bestX = px; bestY = py; }
  }
  return { x: bestX, y: bestY };
}

function highlightPath(pathArr) {
  currentPath = pathArr;   // save for path snapping
  const edgeSet=new Set();
  for(let i=0;i<pathArr.length-1;i++){
    edgeSet.add(`${pathArr[i]}-${pathArr[i+1]}`);
    edgeSet.add(`${pathArr[i+1]}-${pathArr[i]}`);
  }
  document.querySelectorAll('#mapSvg .edge').forEach(l=>{
    l.classList.toggle('path',edgeSet.has(`${l.dataset.from}-${l.dataset.to}`));
  });
  document.querySelectorAll('#mapSvg .node').forEach(c=>{
    const id=c.dataset.id;
    c.classList.remove('start','end');
    if(pathArr.length>0&&id===String(pathArr[0]))                   c.classList.add('start');
    if(pathArr.length>0&&id===String(pathArr[pathArr.length-1]))    c.classList.add('end');
  });
  let len=0;
  for(let i=0;i<pathArr.length-1;i++){
    const a=mapData.nodes[pathArr[i]],b=mapData.nodes[pathArr[i+1]];
    if(a&&b) len+=Math.hypot(a.x-b.x,a.y-b.y);
  }
  document.getElementById('pathLen').textContent  = len.toFixed(2)+' m';
  document.getElementById('pathCount').textContent = pathArr.length;
  const goalB=document.getElementById('goalBadge');
  if(pathArr.length){ goalB.style.display=''; goalB.textContent=`GOAL: ${pathArr[0]} -> ${pathArr[pathArr.length-1]}`; }
}

function previewPath(){
  if(!mapData) return;
  const path=dijkstra(mapData,+document.getElementById('startInput').value,+document.getElementById('endInput').value);
  if(path) highlightPath(path);
}

function dijkstra(mp,src,dst){
  if(!(src in mp.nodes)||!(dst in mp.nodes)) return null;
  const adj={};
  for(const id of Object.keys(mp.nodes)) adj[id]=[];
  for(const e of mp.edges){ adj[e.from].push([e.to,e.length]); }
  const dist={},prev={},vis=new Set();
  for(const id of Object.keys(mp.nodes)) dist[id]=Infinity;
  dist[src]=0;
  const pq=[[0,String(src)]];
  while(pq.length){
    pq.sort((a,b)=>a[0]-b[0]);
    const[d,u]=pq.shift();
    if(vis.has(u)) continue; vis.add(u);
    if(u==dst) break;
    for(const[v,w] of(adj[u]||[])){const nd=d+w; if(nd<dist[v]){dist[v]=nd;prev[v]=u;pq.push([nd,String(v)]);}}
  }
  if(dist[dst]===Infinity) return null;
  const out=[];let u=String(dst);
  while(u!==undefined){out.unshift(+u);u=prev[u];}
  return out;
}

function updateRobot(x, y) {
  if (!svgViewBox) return;
  const m = document.getElementById('robotMarker');
  if (!m) return;
  m.setAttribute('cx', svgViewBox.tx(x));
  m.setAttribute('cy', svgViewBox.ty(y));
  m.style.display = '';
}

// --- Button handlers ----------------------------------------------------------
document.getElementById('previewBtn').onclick = previewPath;

document.getElementById('useNearestBtn').onclick = () => {
  const id = document.getElementById('nearestNodeId').textContent;
  if (id !== '--') document.getElementById('startInput').value = id;
};

// AVVIA: simple lane-following mode (no route plan).
// Cancels any active navigation goal first so waypoint_manager releases enable control.
// 200ms delay ensures waypoint_manager processes the cancel before the enable arrives.
function avvia() {
  if (!goalPub) { alert('ROS non connesso'); return; }
  goalPub.publish(new ROSLIB.Message({ data: [] }));  // cancel any navigation
  setTimeout(() => setDriveEnabled(true), 200);       // wait for waypoint_manager cancel CB
}

// FERMA: stop lane-following and cancel any active navigation.
function ferma() {
  if (goalPub) goalPub.publish(new ROSLIB.Message({ data: [] }));
  setDriveEnabled(false);
}

// NAVIGA: send start→end goal to waypoint_manager.
// If the robot's estimated position differs from "Da" by more than NAV_CONFIRM_DIST,
// shows a confirmation popup so the user can pick the right node or snap and override.

function _doNavigate(start, end) {
  driveEnabled = false;
  syncDriveUI(false);
  goalPub.publish(new ROSLIB.Message({ data: [start, end] }));
}

function _snapAndNavigate(start, end) {
  const refNode = mapData && mapData.nodes[start];
  if (refNode) {
    const { theta, scale } = mapTransform;
    const cosT = Math.cos(theta), sinT = Math.sin(theta);
    mapTransform.tx = lastRawOdom.x - scale * (cosT * refNode.x - sinT * refNode.y);
    mapTransform.ty = lastRawOdom.y - scale * (sinT * refNode.x + cosT * refNode.y);
    publishAndSaveTransform();
  }
  _doNavigate(start, end);
}

function hideNavConfirm() {
  document.getElementById('navConfirmOverlay').style.display = 'none';
}

document.getElementById('startBtn').onclick = () => {
  if (!goalPub) { alert('ROS non connesso'); return; }
  const start = +document.getElementById('startInput').value;
  const end   = +document.getElementById('endInput').value;
  if (isNaN(start) || isNaN(end)) { alert('Nodi Start/End non validi'); return; }

  const refNode = mapData && mapData.nodes[start];
  if (!refNode) { _doNavigate(start, end); return; }

  const mapped      = odomToMap(lastRawOdom.x, lastRawOdom.y);
  const distToStart = Math.hypot(mapped.x - refNode.x, mapped.y - refNode.y);
  const nearest     = nearestMapNode(mapped.x, mapped.y);

  if (distToStart <= NAV_CONFIRM_DIST) {
    // Already close to the declared start — proceed without asking.
    _doNavigate(start, end);
    return;
  }

  // Robot appears far from the entered "Da" node: ask the user.
  const nearId   = nearest ? nearest.id   : '?';
  const nearDist = nearest ? nearest.dist.toFixed(2) : '?';
  document.getElementById('navConfirmMsg').innerHTML =
    `Posizione stimata: nodo <b>${nearId}</b> (a ${nearDist} m dal robot)<br>` +
    `Nodo <em>Da</em> impostato: <b>${start}</b> (a ${distToStart.toFixed(2)} m)`;
  document.getElementById('navConfirmNearest').textContent =
    '► Inizia da nodo ' + nearId + ' (posizione attuale)';
  document.getElementById('navConfirmSnap').textContent =
    '📌 Sono al nodo ' + start + ' — calibra e inizia';
  document.getElementById('navConfirmOverlay').style.display = 'flex';

  document.getElementById('navConfirmNearest').onclick = () => {
    hideNavConfirm();
    document.getElementById('startInput').value = nearId;
    _doNavigate(nearId, end);          // trust existing calibration
  };
  document.getElementById('navConfirmSnap').onclick = () => {
    hideNavConfirm();
    _snapAndNavigate(start, end);      // user confirms they ARE at start — snap then go
  };
  document.getElementById('navConfirmCancel').onclick = hideNavConfirm;
};

// STOP NAV: cancel navigation and stop the robot.
document.getElementById('stopBtn').onclick = () => {
  if (!goalPub) return;
  goalPub.publish(new ROSLIB.Message({ data: [] }));
  setDriveEnabled(false);
};

document.getElementById('calibBtn').onclick = () => {
  const startId = +document.getElementById('startInput').value;
  const refNode = mapData && mapData.nodes[startId];
  if (!refNode) { alert('Mappa non caricata o nodo Start non valido'); return; }

  // Keep existing theta/scale; update only translation to align odom to this node.
  const { theta, scale } = mapTransform;
  const cosT = Math.cos(theta), sinT = Math.sin(theta);
  mapTransform.tx = lastRawOdom.x - scale * (cosT * refNode.x - sinT * refNode.y);
  mapTransform.ty = lastRawOdom.y - scale * (sinT * refNode.x + cosT * refNode.y);

  publishAndSaveTransform();

  const btn = document.getElementById('calibBtn');
  const orig = btn.textContent;
  btn.textContent = '✓ Calibrato';
  btn.style.color = 'var(--green)';
  btn.style.borderColor = 'var(--green)';
  setTimeout(() => {
    btn.textContent = orig;
    btn.style.color = '';
    btn.style.borderColor = '';
  }, 2000);
};

// --- Remap (auto-drive similarity transform) ----------------------------------
// Press Remap: robot drives forward REMAP_NODES hops from Start, measures the
// odom displacement, computes theta+scale+tx+ty non-destructively (mapData.nodes
// stays in original map-frame coordinates; odomToMap() applies the inverse).
const REMAP_NODES = 2;
const REMAP_SPEED = 0.10;   // m/s forward during auto-drive
let remapState      = 0;    // 0 = idle, 1 = driving
let remapDriveTimer = null;
let remapFailTimer  = null;

document.getElementById('remapBtn').onclick = () => {
  if (!mapData) { alert('Mappa non caricata'); return; }
  if (remapState === 1) return;
  const startId = +document.getElementById('startInput').value;
  const nodeA = mapData.nodes[startId];
  if (!nodeA) { alert('Nodo Start non valido'); return; }

  const fwd = findForwardPath(startId, REMAP_NODES);
  if (fwd.path.length <= REMAP_NODES) {
    console.log('Remap path too short:', fwd.path + '; ' + fwd.distance.toFixed(2) + 'm; EndId: ' +  fwd.endId);
    alert('Non ci sono abbastanza nodi in avanti (' + REMAP_NODES + ') dalla Start. Controlla la mappa.');
    return;
  }
  const endId    = fwd.endId;
  const pathDist = fwd.distance;
  const nodeB    = mapData.nodes[endId];
  const odomA    = { x: lastRawOdom.x, y: lastRawOdom.y };
  const nodeAsnap = { x: nodeA.x, y: nodeA.y };
  const nodeBsnap = { x: nodeB.x, y: nodeB.y };
  const timeoutMs = (pathDist / REMAP_SPEED) * 1.5 * 1000;

  if (!remapCmdPub) { alert('ROS non connesso'); return; }

  remapState = 1;

  if (laneEnablePub) laneEnablePub.publish(new ROSLIB.Message({ data: false }));
  const remapRefEl = document.querySelector('#mapSvg .node[data-id="' + startId + '"]');
  if (remapRefEl) remapRefEl.classList.add('remap-ref');

  const btn = document.getElementById('remapBtn');
  btn.textContent = '⏳ Guida…';
  btn.style.background  = 'rgba(246,196,83,.25)';
  btn.style.color       = 'var(--yellow)';
  btn.style.borderColor = 'var(--yellow)';

  remapDriveTimer = setInterval(() => {
    remapCmdPub.publish(new ROSLIB.Message({
      linear: { x: REMAP_SPEED, y: 0, z: 0 },
      angular: { x: 0, y: 0, z: 0 }
    }));
    if (Math.hypot(lastRawOdom.x - odomA.x, lastRawOdom.y - odomA.y) >= pathDist)
      finishRemap(odomA, nodeAsnap, nodeBsnap, false);
  }, 30);

  remapFailTimer = setTimeout(() => {
    if (remapState === 1) finishRemap(odomA, nodeAsnap, nodeBsnap, true);
  }, timeoutMs);
};

function finishRemap(odomA, nodeA, nodeB, timedOut) {
  clearInterval(remapDriveTimer);
  clearTimeout(remapFailTimer);
  remapDriveTimer = remapFailTimer = null;
  remapState = 0;

  remapCmdPub.publish(new ROSLIB.Message({ linear:{x:0,y:0,z:0}, angular:{x:0,y:0,z:0} }));
  if (laneEnablePub) laneEnablePub.publish(new ROSLIB.Message({ data: true }));
  document.querySelectorAll('#mapSvg .node').forEach(el => el.classList.remove('remap-ref'));

  const odomB = { x: lastRawOdom.x, y: lastRawOdom.y };
  const ux = nodeB.x - nodeA.x, uy = nodeB.y - nodeA.y;
  const vx = odomB.x - odomA.x, vy = odomB.y - odomA.y;
  const lenU = Math.hypot(ux, uy), lenV = Math.hypot(vx, vy);

  const btn = document.getElementById('remapBtn');
  if (lenU < 0.01 || lenV < 0.03) {
    btn.textContent = '⚠ Remap fallito';
    btn.style.background  = 'rgba(248,114,114,.18)';
    btn.style.color       = 'var(--red)';
    btn.style.borderColor = 'var(--red)';
    setTimeout(resetRemapBtn, 3000);
    if (timedOut) alert('Timeout Remap: odometria non cambiata. Riposiziona e riprova.');
    else          alert('Remap fallito: distanza percorsa troppo piccola (lenV=' + lenV.toFixed(3) + 'm).');
    return;
  }

  const scale = lenV / lenU;
  const theta = Math.atan2(vy, vx) - Math.atan2(uy, ux);
  const cosT  = Math.cos(theta), sinT = Math.sin(theta);
  mapTransform = {
    theta, scale,
    tx: odomA.x - scale * (cosT * nodeA.x - sinT * nodeA.y),
    ty: odomA.y - scale * (sinT * nodeA.x + cosT * nodeA.y)
  };

  publishAndSaveTransform();
  document.getElementById('mapFrame').textContent = (mapData.frame || 'odom') + ' ✓remapped';

  btn.textContent = timedOut ? '⚠ Timeout (parziale)' : '✓ Remappato';
  btn.style.background  = timedOut ? 'rgba(246,196,83,.25)' : 'rgba(54,211,153,.18)';
  btn.style.color       = timedOut ? 'var(--yellow)' : 'var(--green)';
  btn.style.borderColor = timedOut ? 'var(--yellow)' : 'var(--green)';
  setTimeout(resetRemapBtn, 3000);

  // Stop the car from moving
  setDriveEnabled(false);
}

function resetRemapBtn() {
  const btn = document.getElementById('remapBtn');
  btn.textContent = '🔄 Remap';
  btn.style.background = btn.style.color = btn.style.borderColor = '';
}

function odomToMap(x, y) {
  const { theta, scale, tx, ty } = mapTransform;
  if (scale === 0) return { x, y };
  const cosT = Math.cos(-theta), sinT = Math.sin(-theta);
  const dx = x - tx, dy = y - ty;
  return { x: (cosT * dx - sinT * dy) / scale, y: (sinT * dx + cosT * dy) / scale };
}

function findForwardPath(startId, nHops) {
  const adj = {};
  for (const e of mapData.edges) {
    if (!adj[e.from]) adj[e.from] = [];
    adj[e.from].push({ to: e.to, length: e.length || 0 });
  }
  const path = [startId];
  let cur = startId, totalLength = 0;
  for (let i = 0; i < nHops; i++) {
    const nbrs = adj[cur] || [];
    if (!nbrs.length) break;
    const next = nbrs[0];
    path.push(next.to);
    totalLength += next.length;
    cur = next.to;
  }
  return { path, endId: cur, distance: totalLength };
}

function publishAndSaveTransform() {
  if (remapTransformPub) {
    remapTransformPub.publish(new ROSLIB.Message({
      data: [mapTransform.theta, mapTransform.scale, mapTransform.tx, mapTransform.ty]
    }));
  }
  fetch('/save_remap_params', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(mapTransform)
  }).catch(e => console.error('[remap] save_remap_params failed:', e));
}

function toggleRemapInfo(show) {
  const popup   = document.getElementById('remapInfoPopup');
  const overlay = document.getElementById('remapOverlay');
  const visible = (show === undefined) ? (popup.style.display === 'none') : show;
  popup.style.display = overlay.style.display = visible ? '' : 'none';
}
document.getElementById('remapInfoBtn').onclick = () => toggleRemapInfo();

// --- Init ---------------------------------------------------------------------
loadMap();
