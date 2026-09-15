/**
 * Autonomous Vehicle 3D Real-Time Mission Control & Simulation Engine
 * High-Fidelity WebGL 3D Simulation with Ultra-Realistic Indian Roadways,
 * Crisp High-Visibility Road Markings, 3D Curbs, Milestones, and Continuous AI Physics.
 */

(function () {
  'use strict';

  // --- APPLICATION & PHYSICS STATE ---
  const state = {
    mode: 'auto', // 'auto' | 'manual'
    cameraView: 'chase', // 'chase' | 'cockpit' | 'drone' | 'orbit'
    weather: 'daylight',
    trafficDensity: 'medium',
    isPaused: false,
    simTime: 0,
    distanceTraveled: 0,
    lastPotholeSpawnX: 0,
    lastCrossroadSpawnX: 0,
    stats: {
      collisions: 0,
      nearMisses: 0,
      potholeHits: 0,
      replans: 0,
      speedHistory: [],
      clearanceHistory: []
    },

    // Autonomous Ego Vehicle Physics (Kinematic Bicycle Model)
    ego: {
      x: 0,
      y: 0,
      z: 0.45,
      v: 12.0, // m/s (~43 km/h)
      a: 0.0,
      psi: 0.0, // heading radians
      delta: 0.0, // front steer angle radians
      targetV: 12.5,
      targetDelta: 0.0,
      targetD: 0.0, // Frenet lateral offset
      length: 4.7,
      width: 1.95,
      wheelbase: 2.8,
      maxSteer: 0.62, // ~36 deg
      maxSteerRate: 1.35, // Faster steering actuation for responsive evasion
      maxAccel: 3.2,
      maxDecel: 6.5,
      suspensionJolt: 0.0,
      pitchJolt: 0.0,
      activeManeuver: 'Cruise',
      plannerState: 'NORMAL_DRIVING',
      riskLevel: 'LOW',
      minTTC: 99.9,
      minClearance: 15.0,
      closestHazard: 'Roadway clear',
      gear: 'D', // 'D' | 'R' | 'P'
      isReversing: false,
      reverseTimer: 0.0,
      deadlockTimer: 0.0,
      reverseTargetD: 0.0
    },

    keys: {
      forward: false,
      backward: false,
      left: false,
      right: false
    },

    trafficActors: [],
    potholes: [],
    roadSegments: [],
    roadsideProps: [],
    candidatePathsMesh: null,
    activePathMesh: null
  };

  // --- THREE.JS GLOBALS ---
  let scene, camera, renderer, controls;
  let egoGroup, egoWheels = [], egoHeadlights = [], egoTailLightBar, egoLidarRotor, egoTurnSignals = [];
  let dirLight, hemiLight, terrainMesh, rainParticleSystem, fogObj;
  let radarCanvas, radarCtx, sparklineCanvas, sparklineCtx;
  let asphaltTexture, asphaltBump;

  // --- DOM REFERENCES ---
  const speedVal = document.getElementById('speedVal');
  const targetSpeedVal = document.getElementById('targetSpeedVal');
  const speedNeedle = document.getElementById('speedNeedle');
  const speedArc = document.getElementById('speedArc');
  const accelVal = document.getElementById('accelVal');
  const accelBar = document.getElementById('accelBar');
  const steeringWheelSvg = document.getElementById('steeringWheelSvg');
  const steerDegBadge = document.getElementById('steerDegBadge');
  const headingVal = document.getElementById('headingVal');
  const distanceTraveledVal = document.getElementById('distanceTraveledVal');
  const latDevVal = document.getElementById('latDevVal');
  const ttcVal = document.getElementById('ttcVal');
  const clearanceVal = document.getElementById('clearanceVal');
  const closestHazardVal = document.getElementById('closestHazardVal');
  const riskLevelBadge = document.getElementById('riskLevelBadge');
  const maneuverVal = document.getElementById('maneuverVal');
  const plannerStateVal = document.getElementById('plannerStateVal');
  const latencyVal = document.getElementById('latencyVal');
  const replanCountVal = document.getElementById('replanCountVal');
  const vehicleModeBadge = document.getElementById('vehicleModeBadge');
  const eventTicker = document.getElementById('eventTicker');
  const collisionCountVal = document.getElementById('collisionCountVal');
  const nearMissCountVal = document.getElementById('nearMissCountVal');
  const simTimeVal = document.getElementById('simTimeVal');
  const avgSpeedVal = document.getElementById('avgSpeedVal');

  // --- INITIALIZATION ---
  function init() {
    createProceduralTextures();
    initThreeScene();
    initHUD();
    initEventListeners();
    populateInitialEnvironment();
    addEventLog("Realistic 3D Roadway Loaded: High-visibility lanes, road studs & live Indian traffic.", "text-emerald-400 font-bold");
    animate(0);
  }

  // --- PROCEDURAL TEXTURES (ASPHALT GRAIN & WEAR) ---
  function createProceduralTextures() {
    // Generate high-resolution asphalt canvas texture
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');

    // Base dark charcoal tarmac
    ctx.fillStyle = '#22252a';
    ctx.fillRect(0, 0, 512, 512);

    // Fine mineral aggregate speckles
    const imgData = ctx.getImageData(0, 0, 512, 512);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
      const noise = (Math.random() - 0.5) * 28;
      data[i] = Math.min(255, Math.max(0, data[i] + noise));
      data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + noise));
      data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + noise));
    }
    ctx.putImageData(imgData, 0, 0);

    // Subtle darker tire track bands
    ctx.fillStyle = 'rgba(15, 18, 22, 0.28)';
    ctx.fillRect(0, 70, 512, 110);
    ctx.fillRect(0, 330, 512, 110);

    asphaltTexture = new THREE.CanvasTexture(canvas);
    asphaltTexture.wrapS = THREE.RepeatWrapping;
    asphaltTexture.wrapT = THREE.RepeatWrapping;
    asphaltTexture.repeat.set(16, 2);

    // Contact shadow texture for vehicle grounding
    const shadowCanvas = document.createElement('canvas');
    shadowCanvas.width = 128;
    shadowCanvas.height = 128;
    const sCtx = shadowCanvas.getContext('2d');
    const grad = sCtx.createRadialGradient(64, 64, 15, 64, 64, 60);
    grad.addColorStop(0, 'rgba(0, 0, 0, 0.85)');
    grad.addColorStop(0.5, 'rgba(0, 0, 0, 0.45)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    sCtx.fillStyle = grad;
    sCtx.fillRect(0, 0, 128, 128);
    asphaltBump = new THREE.CanvasTexture(shadowCanvas);
  }

  // --- 3D SCENE & LIGHTING SETUP ---
  function initThreeScene() {
    const container = document.getElementById('threeContainer');
    const w = container.clientWidth;
    const h = container.clientHeight;

    scene = new THREE.Scene();

    // Vibrant Daylight Sky (Light blue with atmospheric horizon)
    scene.background = new THREE.Color(0x87ceeb);

    camera = new THREE.PerspectiveCamera(52, w / h, 0.3, 1400);
    camera.position.set(-13.5, 5.2, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.18;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enabled = false;

    // Dual Lighting: Bright Hemisphere Light + Directional Sunlight
    hemiLight = new THREE.HemisphereLight(0xffffff, 0x475569, 1.05);
    hemiLight.position.set(0, 120, 0);
    scene.add(hemiLight);

    dirLight = new THREE.DirectionalLight(0xfff8ee, 1.45);
    dirLight.position.set(45, 80, 35);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 10;
    dirLight.shadow.camera.far = 280;
    dirLight.shadow.camera.left = -30;
    dirLight.shadow.camera.right = 30;
    dirLight.shadow.camera.top = 80;
    dirLight.shadow.camera.bottom = -30;
    dirLight.shadow.bias = -0.0004;
    scene.add(dirLight);

    // Directional light tracking target
    const lightTarget = new THREE.Object3D();
    scene.add(lightTarget);
    dirLight.target = lightTarget;

    // Atmospheric Distance Fog
    fogObj = new THREE.FogExp2(0xcfe4fa, 0.0032);
    scene.fog = fogObj;

    // Build 3D Models
    createEgoVehicle();
    createEndlessRoadway();
    initTrajectoryMeshes();
    createRainParticles();

    window.addEventListener('resize', onWindowResize);
  }

  function onWindowResize() {
    const container = document.getElementById('threeContainer');
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  // --- REALISTIC 3D SCULPTED EGO VEHICLE ---
  function createEgoVehicle() {
    egoGroup = new THREE.Group();

    // Metallic Electric Crimson Paint Shader
    const carPaintMat = new THREE.MeshStandardMaterial({
      color: 0xc8102e, // Deep Metallic Crimson Red
      metalness: 0.88,
      roughness: 0.18,
      envMapIntensity: 1.3
    });

    const darkTrimMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.65,
      metalness: 0.5
    });

    const chromeMat = new THREE.MeshStandardMaterial({
      color: 0xf1f5f9,
      metalness: 0.95,
      roughness: 0.08
    });

    const glassMat = new THREE.MeshPhysicalMaterial({
      color: 0x38bdf8,
      metalness: 0.1,
      roughness: 0.05,
      transmission: 0.7,
      transparent: true,
      opacity: 0.82
    });

    // 1. Aerodynamic Main Lower Chassis
    const chassisGeom = new THREE.BoxGeometry(4.7, 0.75, 1.95);
    const chassisMesh = new THREE.Mesh(chassisGeom, carPaintMat);
    chassisMesh.position.y = 0.58;
    chassisMesh.castShadow = true;
    chassisMesh.receiveShadow = true;
    egoGroup.add(chassisMesh);

    // 2. Sloped Aerodynamic Front Hood Wedge
    const hoodGeom = new THREE.BoxGeometry(1.6, 0.28, 1.88);
    const hoodMesh = new THREE.Mesh(hoodGeom, carPaintMat);
    hoodMesh.position.set(1.48, 0.82, 0);
    hoodMesh.rotation.z = -0.08;
    hoodMesh.castShadow = true;
    egoGroup.add(hoodMesh);

    // Front Bumper / Radiator Grille
    const grilleGeom = new THREE.BoxGeometry(0.18, 0.42, 1.55);
    const grilleMesh = new THREE.Mesh(grilleGeom, darkTrimMat);
    grilleMesh.position.set(2.32, 0.48, 0);
    egoGroup.add(grilleMesh);

    // Front Aerodynamic Splitter Lip
    const splitterGeom = new THREE.BoxGeometry(0.35, 0.08, 1.96);
    const splitterMesh = new THREE.Mesh(splitterGeom, darkTrimMat);
    splitterMesh.position.set(2.32, 0.22, 0);
    splitterMesh.castShadow = true;
    egoGroup.add(splitterMesh);

    // 3. Curved Cabin Greenhouse (A-pillar slope, roof, fastback C-pillar)
    const cabinGeom = new THREE.BoxGeometry(2.35, 0.72, 1.62);
    const cabinMesh = new THREE.Mesh(cabinGeom, darkTrimMat);
    cabinMesh.position.set(-0.28, 1.28, 0);
    cabinMesh.castShadow = true;
    egoGroup.add(cabinMesh);

    // Raked Front Windshield Glass
    const frontWindshield = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.75, 1.52), glassMat);
    frontWindshield.position.set(0.92, 1.25, 0);
    frontWindshield.rotation.z = -0.44;
    egoGroup.add(frontWindshield);

    // Sloped Rear Window Glass
    const rearGlass = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.65, 1.52), glassMat);
    rearGlass.position.set(-1.48, 1.25, 0);
    rearGlass.rotation.z = 0.42;
    egoGroup.add(rearGlass);

    // Side Windows
    [-0.82, 0.82].forEach(z => {
      const sideWindow = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.54, 0.04), glassMat);
      sideWindow.position.set(-0.25, 1.26, z);
      egoGroup.add(sideWindow);

      // Aerodynamic Side Mirrors
      const mirrorGroup = new THREE.Group();
      mirrorGroup.position.set(0.85, 1.05, z);

      const mirrorArm = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.06, 0.22 * (z > 0 ? 1 : -1)), darkTrimMat);
      mirrorGroup.add(mirrorArm);

      const mirrorBody = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.14, 0.16), carPaintMat);
      mirrorBody.position.set(0, 0, 0.22 * (z > 0 ? 1 : -1));
      mirrorGroup.add(mirrorBody);

      const mirrorGlass = new THREE.Mesh(new THREE.PlaneGeometry(0.2, 0.11), chromeMat);
      mirrorGlass.position.set(-0.12, 0, 0.22 * (z > 0 ? 1 : -1));
      mirrorGlass.rotation.y = -Math.PI / 2;
      mirrorGroup.add(mirrorGlass);

      egoGroup.add(mirrorGroup);
    });

    // 4. Rear Aerodynamic Diffuser & Spoiler Lip
    const rearSpoiler = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.08, 1.75), darkTrimMat);
    rearSpoiler.position.set(-2.28, 0.95, 0);
    egoGroup.add(rearSpoiler);

    const rearDiffuser = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.25, 1.82), darkTrimMat);
    rearDiffuser.position.set(-2.32, 0.32, 0);
    egoGroup.add(rearDiffuser);

    // Dual Chrome Exhaust Tips
    [-0.55, 0.55].forEach(z => {
      const exhaust = new THREE.Mesh(
        new THREE.CylinderGeometry(0.06, 0.06, 0.16, 12).rotateZ(Math.PI / 2),
        chromeMat
      );
      exhaust.position.set(-2.36, 0.28, z);
      egoGroup.add(exhaust);
    });

    // 5. 4 High-Performance Alloy Wheels (Rubber Tread + 5-Spoke Silver Rim + Red Brake Calipers)
    const tireGeom = new THREE.CylinderGeometry(0.38, 0.38, 0.28, 24);
    tireGeom.rotateX(Math.PI / 2);
    const tireMat = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.95 });
    const rimMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.9, roughness: 0.15 });
    const discMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.95, roughness: 0.25 });
    const caliperMat = new THREE.MeshStandardMaterial({ color: 0xef4444, metalness: 0.6, roughness: 0.3 });

    const wheelPositions = [
      { x: 1.4, y: 0.38, z: 0.98, isFront: true },
      { x: 1.4, y: 0.38, z: -0.98, isFront: true },
      { x: -1.4, y: 0.38, z: 0.98, isFront: false },
      { x: -1.4, y: 0.38, z: -0.98, isFront: false }
    ];

    egoWheels = [];
    wheelPositions.forEach(pos => {
      const pivot = new THREE.Group();
      pivot.position.set(pos.x, pos.y, pos.z);

      const wheelGroup = new THREE.Group();

      // Rubber Tire
      const tireMesh = new THREE.Mesh(tireGeom, tireMat);
      tireMesh.castShadow = true;
      wheelGroup.add(tireMesh);

      // Alloy Rim Face
      const rimMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 0.29, 16).rotateX(Math.PI / 2), rimMat);
      wheelGroup.add(rimMesh);

      // Brake Disc
      const discMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.04, 16).rotateX(Math.PI / 2), discMat);
      discMesh.position.z = pos.z > 0 ? -0.06 : 0.06;
      wheelGroup.add(discMesh);

      // Sport Red Brembo Caliper
      const caliperMesh = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.16, 0.06), caliperMat);
      caliperMesh.position.set(0, 0.12, pos.z > 0 ? -0.06 : 0.06);
      pivot.add(caliperMesh); // Attached to pivot, not rotating wheel

      pivot.add(wheelGroup);
      egoGroup.add(pivot);

      egoWheels.push({ pivot, wheelGroup, isFront: pos.isFront });
    });

    // 6. Dual LED Projector Headlights with Light Cones
    egoHeadlights = [];
    [-0.72, 0.72].forEach(zOffset => {
      const spot = new THREE.SpotLight(0xffffff, 4.5, 75, Math.PI / 5.5, 0.32, 1.2);
      spot.position.set(2.4, 0.72, zOffset);
      const target = new THREE.Object3D();
      target.position.set(40.0, 0.1, zOffset);
      egoGroup.add(target);
      spot.target = target;
      spot.castShadow = true;
      egoGroup.add(spot);
      egoHeadlights.push(spot);

      // Glowing Headlight Housing / LED DRL Eyebrow
      const lens = new THREE.Mesh(
        new THREE.BoxGeometry(0.12, 0.14, 0.34),
        new THREE.MeshBasicMaterial({ color: 0xffffff })
      );
      lens.position.set(2.36, 0.72, zOffset);
      egoGroup.add(lens);
    });

    // 7. Full-Width Cyber LED Tail Light Bar
    const tailBarGeom = new THREE.BoxGeometry(0.12, 0.12, 1.82);
    const tailBarMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      emissive: 0xdc2626,
      emissiveIntensity: 1.0
    });
    egoTailLightBar = new THREE.Mesh(tailBarGeom, tailBarMat);
    egoTailLightBar.position.set(-2.35, 0.78, 0);
    egoGroup.add(egoTailLightBar);

    // Amber Turn Signals (Left & Right)
    egoTurnSignals = [];
    [-0.88, 0.88].forEach(zOffset => {
      const turnMesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.14, 0.1, 0.18),
        new THREE.MeshStandardMaterial({ color: 0xf59e0b, emissive: 0x000000 })
      );
      turnMesh.position.set(2.35, 0.65, zOffset);
      egoGroup.add(turnMesh);
      egoTurnSignals.push({ mesh: turnMesh, isRight: zOffset < 0 });
    });

    // 8. Roof ADAS Sensor Pod (360° LiDAR Puck + Spinning Laser Scanner)
    const podBase = new THREE.Mesh(
      new THREE.BoxGeometry(0.8, 0.12, 0.5),
      darkTrimMat
    );
    podBase.position.set(0.0, 1.7, 0.0);
    egoGroup.add(podBase);

    const lidarBase = new THREE.Mesh(
      new THREE.CylinderGeometry(0.18, 0.2, 0.14, 16),
      darkTrimMat
    );
    lidarBase.position.set(0.0, 1.82, 0.0);
    egoGroup.add(lidarBase);

    egoLidarRotor = new THREE.Mesh(
      new THREE.CylinderGeometry(0.16, 0.16, 0.12, 16),
      new THREE.MeshStandardMaterial({ color: 0x06b6d4, emissive: 0x00f2fe, emissiveIntensity: 1.4 })
    );
    egoLidarRotor.position.set(0.0, 1.94, 0.0);
    egoGroup.add(egoLidarRotor);

    // Forward Windshield Dual ADAS Vision Cameras
    const camHousing = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.08, 0.32), darkTrimMat);
    camHousing.position.set(0.72, 1.52, 0);
    egoGroup.add(camHousing);

    // 9. Ground Contact Shadow (Soft Ambient Occlusion)
    const shadowMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(5.2, 2.6).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ map: asphaltBump, transparent: true, opacity: 0.75, depthWrite: false })
    );
    shadowMesh.position.y = 0.02;
    egoGroup.add(shadowMesh);

    scene.add(egoGroup);
  }

  // --- REALISTIC 3D ROADWAY WITH HIGH-VISIBILITY LANES & CAT'S EYES ---
  function createEndlessRoadway() {
    // 1. Infinite Countryside Terrain Plane (Tracks with Ego along X)
    const groundGeom = new THREE.PlaneGeometry(1600, 600);
    groundGeom.rotateX(-Math.PI / 2);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x274e13, // Rich Indian countryside greenery / agricultural landscape
      roughness: 0.95
    });
    terrainMesh = new THREE.Mesh(groundGeom, groundMat);
    terrainMesh.position.set(0, -0.06, 0);
    terrainMesh.receiveShadow = true;
    scene.add(terrainMesh);

    // 2. Procedural Road Segments with Raised Curbs & Crisp Stripes
    state.roadSegments = [];
    const segmentLength = 120;
    const numSegments = 6;
    for (let i = 0; i < numSegments; i++) {
      const seg = buildRealisticRoadSegment(i * segmentLength, segmentLength);
      state.roadSegments.push(seg);
      scene.add(seg);
    }
  }

  function buildRealisticRoadSegment(startX, length) {
    const segGroup = new THREE.Group();
    segGroup.position.x = startX;

    const roadWidth = 9.6; // Wide dual-lane roadway (4.8m per lane)
    const roadHalfWidth = roadWidth / 2;

    // 1. Textured Asphalt Road Pavement
    const roadMat = new THREE.MeshStandardMaterial({
      color: 0x2b2e35, // Realistic dark asphalt
      map: asphaltTexture,
      roughness: 0.85,
      metalness: 0.12
    });
    const roadGeom = new THREE.PlaneGeometry(length, roadWidth);
    roadGeom.rotateX(-Math.PI / 2);
    const roadMesh = new THREE.Mesh(roadGeom, roadMat);
    roadMesh.position.set(length / 2, 0.0, 0);
    roadMesh.receiveShadow = true;
    segGroup.add(roadMesh);

    // 2. Solid High-Visibility White Edge Lines (ELIMINATES Z-FIGHTING: polygonOffset + emissive)
    const edgeLineMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xffffff,
      emissiveIntensity: 0.28,
      roughness: 0.4,
      polygonOffset: true,
      polygonOffsetFactor: -4.0,
      polygonOffsetUnits: -4.0
    });
    const edgeLineGeom = new THREE.PlaneGeometry(length, 0.32);
    edgeLineGeom.rotateX(-Math.PI / 2);

    // Left Solid Edge Line
    const leftEdge = new THREE.Mesh(edgeLineGeom, edgeLineMat);
    leftEdge.position.set(length / 2, 0.03, roadHalfWidth - 0.28);
    segGroup.add(leftEdge);

    // Right Solid Edge Line
    const rightEdge = new THREE.Mesh(edgeLineGeom, edgeLineMat);
    rightEdge.position.set(length / 2, 0.03, -roadHalfWidth + 0.28);
    segGroup.add(rightEdge);

    // 3. Bright Golden Yellow Dashed Centerline Stripes (High Contrast & Visible!)
    const yellowCenterMat = new THREE.MeshStandardMaterial({
      color: 0xfacc15,
      emissive: 0xfacc15,
      emissiveIntensity: 0.32,
      roughness: 0.4,
      polygonOffset: true,
      polygonOffsetFactor: -4.0,
      polygonOffsetUnits: -4.0
    });
    const dashLength = 5.0;
    const dashGap = 5.0;
    const dashGeom = new THREE.PlaneGeometry(dashLength, 0.28);
    dashGeom.rotateX(-Math.PI / 2);

    for (let x = 2; x < length; x += (dashLength + dashGap)) {
      const dash = new THREE.Mesh(dashGeom, yellowCenterMat);
      dash.position.set(x + dashLength / 2, 0.032, 0);
      segGroup.add(dash);
    }

    // 4. 3D Solar Cat's Eyes / RPM Road Studs along Centerline (Amber Reflectors every 5m)
    const studGeom = new THREE.BoxGeometry(0.18, 0.04, 0.18);
    const studMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0xfbbf24,
      emissiveIntensity: 0.85,
      metalness: 0.85,
      roughness: 0.2
    });

    for (let x = 5; x < length; x += 5.0) {
      const stud = new THREE.Mesh(studGeom, studMat);
      stud.position.set(x, 0.045, 0);
      stud.castShadow = true;
      segGroup.add(stud);
    }

    // 5. Zebra Pedestrian Crossing & Speed Markings (placed every 120m)
    const zebraX = 35.0;
    const stripeWidth = 0.5;
    const stripeLen = 4.2;
    const zebraMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xffffff,
      emissiveIntensity: 0.25,
      polygonOffset: true,
      polygonOffsetFactor: -4.0,
      polygonOffsetUnits: -4.0
    });
    const stripeGeom = new THREE.PlaneGeometry(stripeLen, stripeWidth).rotateX(-Math.PI / 2);

    for (let z = -roadHalfWidth + 0.8; z < roadHalfWidth - 0.8; z += 1.0) {
      const stripe = new THREE.Mesh(stripeGeom, zebraMat);
      stripe.position.set(zebraX, 0.035, z);
      segGroup.add(stripe);
    }

    // Red Transverse Rumble Warning Strips before Zebra Crossing
    const rumbleMat = new THREE.MeshStandardMaterial({
      color: 0xdc2626,
      emissive: 0xb91c1c,
      emissiveIntensity: 0.35,
      polygonOffset: true,
      polygonOffsetFactor: -3.0
    });
    for (let i = 0; i < 4; i++) {
      const rumbleGeom = new THREE.PlaneGeometry(0.22, roadWidth - 1.2).rotateX(-Math.PI / 2);
      const rumble = new THREE.Mesh(rumbleGeom, rumbleMat);
      rumble.position.set(zebraX - 12.0 - i * 0.9, 0.036, 0);
      segGroup.add(rumble);
    }

    // 6. Raised Concrete 3D Curbs with Black & Yellow Hazard Stripes (with Crossroad Opening at x=75m)
    const curbHeight = 0.28;
    const curbWidth = 0.48;
    const curbSectionLen = 3.0;

    const curbMatYellow = new THREE.MeshStandardMaterial({ color: 0xfacc15, roughness: 0.75 });
    const curbMatBlack = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.85 });

    for (let x = 0; x < length; x += curbSectionLen) {
      // Leave opening for the crossroad intersection between x=69 and x=81
      if (x >= 69.0 && x < 81.0) continue;

      const isYellow = Math.floor(x / curbSectionLen) % 2 === 0;
      const mat = isYellow ? curbMatYellow : curbMatBlack;
      const curbBoxGeom = new THREE.BoxGeometry(curbSectionLen, curbHeight, curbWidth);

      // Left Raised Curb
      const leftCurb = new THREE.Mesh(curbBoxGeom, mat);
      leftCurb.position.set(x + curbSectionLen / 2, curbHeight / 2, roadHalfWidth + curbWidth / 2);
      leftCurb.castShadow = true;
      leftCurb.receiveShadow = true;
      segGroup.add(leftCurb);

      // Right Raised Curb
      const rightCurb = new THREE.Mesh(curbBoxGeom, mat);
      rightCurb.position.set(x + curbSectionLen / 2, curbHeight / 2, -roadHalfWidth - curbWidth / 2);
      rightCurb.castShadow = true;
      rightCurb.receiveShadow = true;
      segGroup.add(rightCurb);
    }

    // 7. Red Dirt / Gravel Shoulders Outside Curbs
    const shoulderMat = new THREE.MeshStandardMaterial({ color: 0x854d0e, roughness: 0.96 });
    const shoulderGeom = new THREE.PlaneGeometry(length, 4.5).rotateX(-Math.PI / 2);

    const leftShoulder = new THREE.Mesh(shoulderGeom, shoulderMat);
    leftShoulder.position.set(length / 2, -0.02, roadHalfWidth + curbWidth + 2.25);
    leftShoulder.receiveShadow = true;
    segGroup.add(leftShoulder);

    const rightShoulder = new THREE.Mesh(shoulderGeom, shoulderMat);
    rightShoulder.position.set(length / 2, -0.02, -roadHalfWidth - curbWidth - 2.25);
    rightShoulder.receiveShadow = true;
    segGroup.add(rightShoulder);

    // 8. 3D Crossroad Lateral Intersection (Extends 24m across left and right)
    const crossroadX = 75.0;
    const crossroadW = 9.0;
    const crossroadL = 48.0;
    const crossroadGeom = new THREE.PlaneGeometry(crossroadW, crossroadL).rotateX(-Math.PI / 2);
    const crossroadMesh = new THREE.Mesh(crossroadGeom, roadMat);
    crossroadMesh.position.set(crossroadX, 0.006, 0);
    crossroadMesh.receiveShadow = true;
    segGroup.add(crossroadMesh);

    // Junction Stop / Yield Dashed Markings
    const stopLineMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xffffff,
      emissiveIntensity: 0.3,
      polygonOffset: true,
      polygonOffsetFactor: -3.0
    });
    const leftStop = new THREE.Mesh(new THREE.PlaneGeometry(crossroadW, 0.4).rotateX(-Math.PI / 2), stopLineMat);
    leftStop.position.set(crossroadX, 0.035, roadHalfWidth + 0.2);
    segGroup.add(leftStop);

    const rightStop = new THREE.Mesh(new THREE.PlaneGeometry(crossroadW, 0.4).rotateX(-Math.PI / 2), stopLineMat);
    rightStop.position.set(crossroadX, 0.035, -roadHalfWidth - 0.2);
    segGroup.add(rightStop);

    // Highway Stop Line (White stop bar where ego vehicle halts to yield right-of-way)
    const highwayStop = new THREE.Mesh(new THREE.PlaneGeometry(0.5, roadWidth).rotateX(-Math.PI / 2), stopLineMat);
    highwayStop.position.set(crossroadX - crossroadW / 2 - 1.2, 0.035, 0);
    segGroup.add(highwayStop);

    // Corner Junction Posts with Amber Flashing Solar Blinkers
    const cornerPosts = [
      { x: crossroadX - crossroadW / 2 - 0.4, z: roadHalfWidth + 0.6 },
      { x: crossroadX + crossroadW / 2 + 0.4, z: roadHalfWidth + 0.6 },
      { x: crossroadX - crossroadW / 2 - 0.4, z: -roadHalfWidth - 0.6 },
      { x: crossroadX + crossroadW / 2 + 0.4, z: -roadHalfWidth - 0.6 }
    ];
    cornerPosts.forEach(cp => {
      const post = new THREE.Mesh(
        new THREE.CylinderGeometry(0.12, 0.12, 1.0, 10),
        new THREE.MeshStandardMaterial({ color: 0xfacc15, roughness: 0.6 })
      );
      post.position.set(cp.x, 0.5, cp.z);
      post.castShadow = true;
      segGroup.add(post);

      const solarBlinker = new THREE.Mesh(
        new THREE.SphereGeometry(0.16, 8, 8),
        new THREE.MeshStandardMaterial({ color: 0xf59e0b, emissive: 0xfbbf24, emissiveIntensity: 1.4 })
      );
      solarBlinker.position.set(cp.x, 1.08, cp.z);
      segGroup.add(solarBlinker);
    });

    // 9. Authentic Roadside Elements (Milestones, Streetlights, Trees, Overhead Signs)
    addIndianRoadsideScenery(segGroup, length, roadWidth, roadHalfWidth);

    return segGroup;
  }

  function addIndianRoadsideScenery(group, length, roadWidth, roadHalfWidth) {
    const treeMatTrunk = new THREE.MeshStandardMaterial({ color: 0x45220c, roughness: 0.9 });
    const treeMatLeaves = new THREE.MeshStandardMaterial({ color: 0x15803d, roughness: 0.75 });
    const poleMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.8, roughness: 0.3 });
    const lampGlowMat = new THREE.MeshBasicMaterial({ color: 0xfef08a });

    for (let x = 15; x < length; x += 30) {
      // Don't place trees in the middle of crossroad intersection
      if (x >= 65.0 && x <= 85.0) continue;

      // Lush Trees on Both Sides
      [-12.0, 12.0].forEach(zSide => {
        if (Math.random() > 0.25) {
          const treeGroup = new THREE.Group();
          treeGroup.position.set(x + (Math.random() * 6 - 3), 0, zSide + (Math.random() * 4 - 2));

          const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.45, 4.2, 8), treeMatTrunk);
          trunk.position.y = 2.1;
          trunk.castShadow = true;
          treeGroup.add(trunk);

          const leaves = new THREE.Mesh(new THREE.DodecahedronGeometry(2.6 + Math.random() * 0.9, 1), treeMatLeaves);
          leaves.position.y = 5.2;
          leaves.castShadow = true;
          treeGroup.add(leaves);

          group.add(treeGroup);
        }
      });

      // Cobra-Head Highway Streetlights every 60m
      if (x % 60 === 0 && (x < 65.0 || x > 85.0)) {
        const lightPole = new THREE.Group();
        lightPole.position.set(x, 0, roadHalfWidth + 1.2);

        const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.16, 8.5, 12), poleMat);
        pole.position.y = 4.25;
        pole.castShadow = true;
        lightPole.add(pole);

        const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 2.8, 8).rotateZ(-Math.PI / 4), poleMat);
        arm.position.set(0.95, 8.8, 0);
        lightPole.add(arm);

        const lampFixture = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.15, 0.25), poleMat);
        lampFixture.position.set(1.95, 9.7, 0);
        lightPole.add(lampFixture);

        const bulb = new THREE.Mesh(new THREE.PlaneGeometry(0.4, 0.2).rotateX(Math.PI / 2), lampGlowMat);
        bulb.position.set(1.95, 9.62, 0);
        lightPole.add(bulb);

        group.add(lightPole);
      }

      // Traditional Indian Yellow-Dome Kilometre Milestone Markers every 40m
      if (x % 40 === 0 && (x < 65.0 || x > 85.0)) {
        const stoneGroup = new THREE.Group();
        stoneGroup.position.set(x, 0, roadHalfWidth + 0.85);

        const whiteBody = new THREE.Mesh(
          new THREE.CylinderGeometry(0.28, 0.28, 0.85, 16),
          new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.9 })
        );
        whiteBody.position.y = 0.42;
        whiteBody.castShadow = true;
        stoneGroup.add(whiteBody);

        const yellowDome = new THREE.Mesh(
          new THREE.SphereGeometry(0.28, 16, 12, 0, Math.PI * 2, 0, Math.PI / 2),
          new THREE.MeshStandardMaterial({ color: 0xfacc15, roughness: 0.8 })
        );
        yellowDome.position.y = 0.85;
        stoneGroup.add(yellowDome);

        group.add(stoneGroup);
      }
    }

    // Overhead Highway Sign Gantry spanning over the road
    const gantryGroup = new THREE.Group();
    gantryGroup.position.set(length * 0.75, 0, 0);

    const gantryPoleLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.18, 7.5, 8), poleMat);
    gantryPoleLeft.position.set(0, 3.75, roadHalfWidth + 1.2);
    gantryPoleLeft.castShadow = true;
    gantryGroup.add(gantryPoleLeft);

    const gantryPoleRight = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.18, 7.5, 8), poleMat);
    gantryPoleRight.position.set(0, 3.75, -roadHalfWidth - 1.2);
    gantryPoleRight.castShadow = true;
    gantryGroup.add(gantryPoleRight);

    // Cross Truss
    const truss = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.6, roadWidth + 2.8), poleMat);
    truss.position.set(0, 7.2, 0);
    truss.castShadow = true;
    gantryGroup.add(truss);

    // Green Signboard ("NH-48 EXPRESSWAY: ADAPTIVE AUTONOMOUS TESTBED")
    const signMat = new THREE.MeshStandardMaterial({ color: 0x15803d, roughness: 0.5 });
    const signBoard = new THREE.Mesh(new THREE.BoxGeometry(0.12, 1.4, 5.8), signMat);
    signBoard.position.set(-0.35, 7.0, 0);
    gantryGroup.add(signBoard);

    group.add(gantryGroup);
  }

  // --- RECYCLE ROAD SEGMENTS (INFINITE HIGHWAY) ---
  function updateRoadSegments() {
    const egoX = state.ego.x;
    const segLen = 120;
    const totalLen = segLen * state.roadSegments.length;
    for (const seg of state.roadSegments) {
      while (seg.position.x < egoX - segLen * 1.2) {
        seg.position.x += totalLen;
      }
      while (seg.position.x > egoX + totalLen - segLen * 0.8) {
        seg.position.x -= totalLen;
      }
    }
  }

  // Helper to locate the exact upcoming crossroad intersection coordinate along the highway
  function getUpcomingCrossroadX(minDistAhead = 15) {
    const egoX = state.ego.x;
    let closestX = Infinity;
    for (const seg of state.roadSegments) {
      const junctionX = seg.position.x + 75.0;
      if (junctionX > egoX + minDistAhead && junctionX < closestX) {
        closestX = junctionX;
      }
    }
    if (!isFinite(closestX)) {
      const k = Math.floor((egoX + minDistAhead - 75.0) / 120.0) + 1;
      closestX = k * 120.0 + 75.0;
    }
    return closestX;
  }

  // --- DYNAMIC CROSSROAD TRAFFIC INJECTOR ---
  function spawnCrossroadActor(targetCrossroadX = null) {
    const ego = state.ego;
    const crossX = (typeof targetCrossroadX === 'number') ? targetCrossroadX : getUpcomingCrossroadX(20);
    const fromLeft = Math.random() > 0.5;
    const startY = fromLeft ? 22.0 : -22.0;
    const targetVy = fromLeft ? -4.8 : 4.8;
    const heading = fromLeft ? -Math.PI / 2 : Math.PI / 2; // Facing perpendicular across highway

    // Two-way cross street lanes (9.0m wide asphalt centered at crossX):
    // Left-to-right (fromLeft, heading -y): Travels on its left lane -> crossX - 2.0
    // Right-to-left (fromRight, heading +y): Travels on its left lane -> crossX + 2.0
    const spawnX = fromLeft ? (crossX - 2.0) : (crossX + 2.0);

    const types = ['auto_rickshaw', 'auto_rickshaw', 'truck', 'motorcycle'];
    const type = types[Math.floor(Math.random() * types.length)];

    let group = new THREE.Group();
    let length = 2.8, width = 1.45;
    if (type === 'auto_rickshaw') {
      group = buildAutoRickshawModel();
      length = 2.8; width = 1.45;
    } else if (type === 'truck') {
      group = buildTataTruckModel();
      length = 6.5; width = 2.5;
    } else if (type === 'motorcycle') {
      group = buildMotorcycleModel();
      length = 2.1; width = 0.85;
    }

    group.position.set(spawnX, 0, startY);
    group.rotation.y = -heading;
    scene.add(group);

    state.trafficActors.push({
      id: Math.random(),
      type: type,
      mesh: group,
      x: spawnX,
      y: startY,
      crossroadX: crossX,
      speed: 0.0,
      vx: 0.0,
      vy: targetVy,
      nominalVy: targetVy,
      targetVy: targetVy,
      heading: heading,
      targetHeading: heading,
      nominalSpeed: 0.0,
      targetSpeed: 0.0,
      desiredY: fromLeft ? -24.0 : 24.0,
      nominalY: startY,
      length: length,
      width: width,
      behaviorState: 'approaching_stop_line',
      behaviorTimer: 8.0,
      walkPhase: 0,
      walkSpeed: Math.abs(targetVy),
      weavePhase: 0,
      originY: startY,
      hasCollided: false,
      stoppedForObstacle: false,
      isCrossroadActor: true
    });
  }

  // --- TRAJECTORY VISUAL RIBBONS ---
  function initTrajectoryMeshes() {
    // Active Planned Path (Glowing Neon Emerald Line)
    const activeGeom = new THREE.BufferGeometry();
    const activeMat = new THREE.LineBasicMaterial({
      color: 0x10b981,
      linewidth: 6
    });
    state.activePathMesh = new THREE.Line(activeGeom, activeMat);
    state.activePathMesh.frustumCulled = false;
    scene.add(state.activePathMesh);

    // Candidate Trajectory Fan (Cyan)
    const candGeom = new THREE.BufferGeometry();
    const candMat = new THREE.LineBasicMaterial({
      color: 0x00f2fe,
      transparent: true,
      opacity: 0.38
    });
    state.candidatePathsMesh = new THREE.LineSegments(candGeom, candMat);
    state.candidatePathsMesh.frustumCulled = false;
    scene.add(state.candidatePathsMesh);
  }

  // --- 3D RAIN PARTICLE SYSTEM ---
  function createRainParticles() {
    const count = 1600;
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 160;
      positions[i + 1] = Math.random() * 50;
      positions[i + 2] = (Math.random() - 0.5) * 70;
    }
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const mat = new THREE.PointsMaterial({
      color: 0xd0e2ec,
      size: 0.35,
      transparent: true,
      opacity: 0.65
    });
    rainParticleSystem = new THREE.Points(geom, mat);
    rainParticleSystem.visible = false;
    scene.add(rainParticleSystem);
  }

  // --- 3D INDIAN TRAFFIC MODELS ---
  function spawnActor(type, aheadDist, lateralY, speed, customParams = {}) {
    const ego = state.ego;
    const spawnX = ego.x + aheadDist;

    let group = new THREE.Group();
    let length = 2.0, width = 1.0;

    if (type === 'auto_rickshaw') {
      group = buildAutoRickshawModel();
      length = 2.8;
      width = 1.45;
    } else if (type === 'cattle') {
      group = buildDesiCowModel(customParams.color);
      length = 2.4;
      width = 1.1;
    } else if (type === 'motorcycle') {
      group = buildMotorcycleModel();
      length = 2.1;
      width = 0.85;
    } else if (type === 'truck') {
      group = buildTataTruckModel();
      length = 7.8;
      width = 2.65;
    }

    group.position.set(spawnX, 0, lateralY);
    scene.add(group);

    // Dynamic initial movement vectors and behavior
    let vx = speed;
    let vy = 0;
    let heading = speed >= 0 ? 0 : Math.PI;
    let behaviorState = customParams.mode || 'normal';
    const walkSpeed = customParams.walkSpeed || (0.55 + Math.random() * 0.2);

    if (type === 'cattle') {
      // Natural Indian cattle behaviors: calm, deliberate crossing with stable direction commitment
      if (!customParams.mode) {
        if (lateralY < -2.6) {
          // Starting from right shoulder: cross towards left or graze calmly on shoulder
          behaviorState = Math.random() > 0.35 ? 'crossing_to_left' : 'grazing_shoulder';
        } else if (lateralY > 2.6) {
          // Starting from left shoulder: cross towards right or graze calmly on shoulder
          behaviorState = Math.random() > 0.35 ? 'crossing_to_right' : 'grazing_shoulder';
        } else {
          // Starting in road corridor: commit to crossing toward nearest/opposite shoulder
          behaviorState = lateralY >= 0 ? 'crossing_to_left' : 'crossing_to_right';
        }
      }

      if (behaviorState === 'crossing_to_left') {
        vy = walkSpeed;
        vx = 0.04;
        heading = Math.PI / 2; // Facing across toward left
      } else if (behaviorState === 'crossing_to_right') {
        vy = -walkSpeed;
        vx = 0.04;
        heading = -Math.PI / 2; // Facing across toward right
      } else if (behaviorState === 'grazing_shoulder') {
        vy = 0.0;
        vx = 0.03;
        heading = 0.0;
      } else if (behaviorState === 'standing_calm') {
        vy = 0;
        vx = 0;
        heading = Math.PI / 2;
      }
    } else if (type === 'motorcycle') {
      behaviorState = customParams.mode || 'filtering';
    }

    group.rotation.y = -heading;

    state.trafficActors.push({
      id: Math.random(),
      type: type,
      mesh: group,
      x: spawnX,
      y: lateralY,
      speed: speed,
      vx: vx,
      vy: vy,
      heading: heading,
      targetHeading: heading,
      nominalSpeed: speed,
      targetSpeed: speed,
      desiredY: lateralY,
      nominalY: lateralY,
      length: length,
      width: width,
      behaviorState: behaviorState,
      behaviorTimer: 3.5 + Math.random() * 4.5,
      walkPhase: Math.random() * 10,
      walkSpeed: walkSpeed,
      weavePhase: Math.random() * Math.PI * 2,
      originY: lateralY,
      hasCollided: false,
      stoppedForObstacle: false
    });
  }

  // 3-Wheeled Bajaj Auto-Rickshaw (Yellow Canopy Roof + Emerald Body)
  function buildAutoRickshawModel() {
    const rick = new THREE.Group();
    const greenMat = new THREE.MeshStandardMaterial({ color: 0x059669, metalness: 0.6, roughness: 0.4 });
    const yellowMat = new THREE.MeshStandardMaterial({ color: 0xfacc15, metalness: 0.3, roughness: 0.5 });
    const darkMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.85 });
    const chromeMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.9, roughness: 0.1 });

    // Lower Green Metal Cabin
    const body = new THREE.Mesh(new THREE.BoxGeometry(2.7, 0.7, 1.45), greenMat);
    body.position.y = 0.65;
    body.castShadow = true;
    rick.add(body);

    // Front Tapered Nose
    const nose = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.55, 1.1), greenMat);
    nose.position.set(1.4, 0.62, 0);
    nose.castShadow = true;
    rick.add(nose);

    // Iconic Yellow Canopy Roof
    const roof = new THREE.Mesh(new THREE.BoxGeometry(2.5, 0.65, 1.48), yellowMat);
    roof.position.set(0.1, 1.35, 0);
    roof.castShadow = true;
    rick.add(roof);

    // Front Windshield Frame
    const windshield = new THREE.Mesh(
      new THREE.BoxGeometry(0.05, 0.55, 1.25),
      new THREE.MeshPhysicalMaterial({ color: 0x93c5fd, transmission: 0.7, transparent: true, opacity: 0.85 })
    );
    windshield.position.set(1.36, 1.22, 0);
    rick.add(windshield);

    // Wheels (1 Front + 2 Rear)
    const tireGeom = new THREE.CylinderGeometry(0.26, 0.26, 0.18, 16).rotateX(Math.PI / 2);
    const frontWheel = new THREE.Mesh(tireGeom, darkMat);
    frontWheel.position.set(1.45, 0.26, 0);
    frontWheel.castShadow = true;
    rick.add(frontWheel);

    [-0.68, 0.68].forEach(z => {
      const rearWheel = new THREE.Mesh(tireGeom, darkMat);
      rearWheel.position.set(-0.75, 0.26, z);
      rearWheel.castShadow = true;
      rick.add(rearWheel);
    });

    // Single Round Chrome Headlight
    const headlamp = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 0.08, 16).rotateZ(Math.PI / 2), chromeMat);
    headlamp.position.set(1.82, 0.68, 0);
    rick.add(headlamp);

    return rick;
  }

  // Indian Desi Humped Cow (Zebu) with Articulated 4-Leg Hip Pivots & Animated Gait
  function buildDesiCowModel(colorHex) {
    const cow = new THREE.Group();
    const hideColor = colorHex || (Math.random() > 0.6 ? 0xd6d3d1 : (Math.random() > 0.5 ? 0x854d0e : 0x292524));
    const hideMat = new THREE.MeshStandardMaterial({ color: hideColor, roughness: 0.95 });
    const hornMat = new THREE.MeshStandardMaterial({ color: 0x292524, roughness: 0.7 });
    const hoofMat = new THREE.MeshStandardMaterial({ color: 0x1c1917, roughness: 0.9 });

    // Torso Barrel Body
    const body = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.95, 0.88), hideMat);
    body.position.y = 1.15;
    body.castShadow = true;
    cow.add(body);

    // Iconic Zebu Dorsal Hump on Shoulders
    const hump = new THREE.Mesh(new THREE.DodecahedronGeometry(0.42, 1), hideMat);
    hump.position.set(0.45, 1.75, 0);
    hump.scale.set(1.1, 1.3, 0.85);
    hump.castShadow = true;
    cow.add(hump);

    // Articulated Neck & Head Pivot for Looking/Alert Behavior
    const neckPivot = new THREE.Group();
    neckPivot.position.set(0.95, 1.35, 0);

    const neck = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.65, 0.55), hideMat);
    neck.position.set(0.1, 0.1, 0);
    neck.rotation.z = -0.32;
    neckPivot.add(neck);

    const head = new THREE.Mesh(new THREE.BoxGeometry(0.65, 0.48, 0.52), hideMat);
    head.position.set(0.4, 0.1, 0);
    head.castShadow = true;
    neckPivot.add(head);

    // Curved Horns & Droopy Ears
    [-0.22, 0.22].forEach(z => {
      const horn = new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.45, 8), hornMat);
      horn.position.set(0.3, 0.45, z);
      horn.rotation.z = -0.3;
      horn.rotation.x = z > 0 ? 0.35 : -0.35;
      neckPivot.add(horn);

      const ear = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.08, 0.32), hideMat);
      ear.position.set(0.24, 0.18, z * 1.3);
      ear.rotation.x = z > 0 ? 0.45 : -0.45;
      neckPivot.add(ear);
    });

    cow.add(neckPivot);
    cow.userData.neckPivot = neckPivot;

    // 4 Articulated Legs with Hip Pivots for natural quadruped walk cycles
    const legPositions = [
      { x: 0.65, z: 0.32 },   // Front Left (0)
      { x: 0.65, z: -0.32 },  // Front Right (1)
      { x: -0.65, z: 0.32 },  // Rear Left (2)
      { x: -0.65, z: -0.32 }  // Rear Right (3)
    ];
    const legPivots = [];
    legPositions.forEach(pos => {
      const hipPivot = new THREE.Group();
      hipPivot.position.set(pos.x, 0.90, pos.z);

      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.78, 0.18), hideMat);
      leg.position.y = -0.38;
      leg.castShadow = true;
      hipPivot.add(leg);

      const hoof = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.12, 0.2), hoofMat);
      hoof.position.y = -0.78;
      hipPivot.add(hoof);

      cow.add(hipPivot);
      legPivots.push(hipPivot);
    });
    cow.userData.legPivots = legPivots;

    // Tail with swish pivot
    const tailPivot = new THREE.Group();
    tailPivot.position.set(-0.95, 1.45, 0);
    const tail = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, 0.85, 6), hideMat);
    tail.position.set(0, -0.42, 0);
    tail.rotation.z = 0.25;
    tailPivot.add(tail);
    cow.add(tailPivot);
    cow.userData.tailPivot = tailPivot;

    return cow;
  }

  // Tata 1613 Heavy Cargo Freight Truck (Vibrant Indian Highway Lorry)
  function buildTataTruckModel() {
    const truck = new THREE.Group();
    const cabMat = new THREE.MeshStandardMaterial({ color: 0x0284c7, metalness: 0.6, roughness: 0.4 }); // Blue Cab
    const cargoMat = new THREE.MeshStandardMaterial({ color: 0xb45309, roughness: 0.8 }); // Wooden brown cargo
    const bumperMat = new THREE.MeshStandardMaterial({ color: 0xdc2626, roughness: 0.6 }); // Red chevron bumper
    const darkMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.9 });

    // Front Truck Cab
    const cab = new THREE.Mesh(new THREE.BoxGeometry(2.4, 2.2, 2.5), cabMat);
    cab.position.set(2.4, 1.7, 0);
    cab.castShadow = true;
    truck.add(cab);

    // Windshield
    const windshield = new THREE.Mesh(
      new THREE.BoxGeometry(0.06, 0.85, 2.3),
      new THREE.MeshPhysicalMaterial({ color: 0x93c5fd, transmission: 0.7, transparent: true, opacity: 0.85 })
    );
    windshield.position.set(3.62, 2.1, 0);
    truck.add(windshield);

    // Colorful Overhead Sun Visor
    const visor = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.25, 2.54), bumperMat);
    visor.position.set(3.5, 2.7, 0);
    truck.add(visor);

    // Heavy Wooden Slatted Cargo Bed
    const cargo = new THREE.Mesh(new THREE.BoxGeometry(5.2, 2.2, 2.65), cargoMat);
    cargo.position.set(-1.4, 1.85, 0);
    cargo.castShadow = true;
    truck.add(cargo);

    // Red Front Bumper
    const bumper = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.5, 2.6), bumperMat);
    bumper.position.set(3.65, 0.55, 0);
    bumper.castShadow = true;
    truck.add(bumper);

    // 6 Large Commercial Wheels
    const tireGeom = new THREE.CylinderGeometry(0.55, 0.55, 0.38, 20).rotateX(Math.PI / 2);
    [
      { x: 2.3, z: 1.25 }, { x: 2.3, z: -1.25 },
      { x: -1.1, z: 1.25 }, { x: -1.1, z: -1.25 },
      { x: -2.6, z: 1.25 }, { x: -2.6, z: -1.25 }
    ].forEach(p => {
      const wheel = new THREE.Mesh(tireGeom, darkMat);
      wheel.position.set(p.x, 0.55, p.z);
      wheel.castShadow = true;
      truck.add(wheel);
    });

    return truck;
  }

  // Motorcycle & Rider with Dynamic Roll Pivot for Banking
  function buildMotorcycleModel() {
    const bike = new THREE.Group();
    const rollGroup = new THREE.Group(); // Inner roll group for motorcycle lean/banking
    bike.add(rollGroup);
    bike.userData.rollGroup = rollGroup;

    const frameMat = new THREE.MeshStandardMaterial({ color: 0x9333ea, metalness: 0.7, roughness: 0.3 });
    const tireMat = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.95 });
    const riderMat = new THREE.MeshStandardMaterial({ color: 0x1e3a8a, roughness: 0.6 });
    const helmetMat = new THREE.MeshStandardMaterial({ color: 0xfacc15, roughness: 0.4 });

    // Wheels
    const tireGeom = new THREE.CylinderGeometry(0.34, 0.34, 0.12, 16).rotateX(Math.PI / 2);
    const frontWheel = new THREE.Mesh(tireGeom, tireMat);
    frontWheel.position.set(0.85, 0.34, 0);
    frontWheel.castShadow = true;
    rollGroup.add(frontWheel);

    const rearWheel = new THREE.Mesh(tireGeom, tireMat);
    rearWheel.position.set(-0.85, 0.34, 0);
    rearWheel.castShadow = true;
    rollGroup.add(rearWheel);

    // Fuel Tank & Engine Frame
    const tank = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.35, 0.38), frameMat);
    tank.position.set(0.2, 0.75, 0);
    rollGroup.add(tank);

    // 3D Rider
    const riderBody = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.75, 0.45), riderMat);
    riderBody.position.set(-0.15, 1.15, 0);
    riderBody.rotation.z = -0.22;
    riderBody.castShadow = true;
    rollGroup.add(riderBody);

    // Helmet
    const helmet = new THREE.Mesh(new THREE.SphereGeometry(0.2, 14, 12), helmetMat);
    helmet.position.set(0.02, 1.68, 0);
    helmet.castShadow = true;
    rollGroup.add(helmet);

    return bike;
  }

  // Realistic Recessed 3D Pothole on Road Surface
  function spawnPothole(aheadDist, lateralY, radius) {
    const ego = state.ego;
    const px = ego.x + aheadDist;

    const potGroup = new THREE.Group();
    potGroup.position.set(px, 0.02, lateralY);

    // Broken dark asphalt crater
    const craterGeom = new THREE.CylinderGeometry(radius, radius * 0.72, 0.09, 18);
    const craterMat = new THREE.MeshStandardMaterial({ color: 0x0a0c10, roughness: 0.98 });
    const craterMesh = new THREE.Mesh(craterGeom, craterMat);
    craterMesh.position.y = -0.045;
    potGroup.add(craterMesh);

    // High-visibility yellow/red caution warning ring
    const ringGeom = new THREE.RingGeometry(radius * 0.92, radius * 1.15, 18).rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0xef4444, side: THREE.DoubleSide });
    const ringMesh = new THREE.Mesh(ringGeom, ringMat);
    ringMesh.position.y = 0.02;
    potGroup.add(ringMesh);

    // Inner gravel & asphalt fracture aggregate
    const aggregateGeom = new THREE.DodecahedronGeometry(radius * 0.35, 0);
    const aggregateMat = new THREE.MeshStandardMaterial({ color: 0x1f2937, roughness: 0.95 });
    const aggregate = new THREE.Mesh(aggregateGeom, aggregateMat);
    aggregate.position.set(0.06, -0.01, 0.04);
    aggregate.scale.set(1.0, 0.25, 0.8);
    potGroup.add(aggregate);

    scene.add(potGroup);
    state.potholes.push({ mesh: potGroup, x: px, y: lateralY, radius: radius, hasHit: false });
  }

  function populateInitialEnvironment() {
    // Clear old traffic & potholes
    state.trafficActors.forEach(a => scene.remove(a.mesh));
    state.trafficActors = [];
    state.potholes.forEach(p => scene.remove(p.mesh));
    state.potholes = [];
    state.lastPotholeSpawnX = state.ego.x;

    // Pre-populate initial traffic ahead across diverse lane positions
    spawnActor('truck', 46, -1.8, 7.0);
    spawnActor('auto_rickshaw', 30, 1.6, 5.8);
    // Cattle starting on left shoulder with natural crossing behavior toward right
    spawnActor('cattle', 58, 3.8, 0.65, { mode: 'crossing_to_right' });
    // Oncoming motorcycle with dynamic filtering
    spawnActor('motorcycle', 88, 2.0, -8.5);
    // Realistic surface potholes ahead
    spawnPothole(26, 0.35, 0.75);
    spawnPothole(68, -1.5, 0.65);
  }

  // --- CLOSED-LOOP PHYSICS & LATERAL PLANNING ENGINE ---
  function updateSimulationPhysics(dt) {
    if (state.isPaused) return;

    state.simTime += dt;
    const ego = state.ego;

    if (state.mode === 'auto') {
      runAdaptiveLatticePlanner(dt);
    } else if (state.mode === 'manual') {
      runManualController(dt);
    } else if (state.mode === 'reverse') {
      runReverseController(dt);
    }

    // Kinematic Bicycle Model Update (Supports forward cruise & reverse backing up to -4.5 m/s)
    ego.v = Math.max(-4.5, Math.min(22.0, ego.v + ego.a * dt));
    const steerDiff = ego.targetDelta - ego.delta;
    ego.delta += Math.sign(steerDiff) * Math.min(Math.abs(steerDiff), ego.maxSteerRate * dt);
    ego.delta = Math.max(-ego.maxSteer, Math.min(ego.maxSteer, ego.delta));

    // Update active gear
    if (ego.v < -0.15 || state.mode === 'reverse' || ego.isReversing) {
      ego.gear = 'R';
    } else if (Math.abs(ego.v) <= 0.15 && Math.abs(ego.a) < 0.2) {
      ego.gear = state.mode === 'manual' ? 'P' : 'D';
    } else {
      ego.gear = 'D';
    }

    // Heading yaw rate dPsi = v/L * tan(delta)
    const psiDot = (ego.v / ego.wheelbase) * Math.tan(ego.delta);
    ego.psi += psiDot * dt;

    // Longitudinal and Lateral displacement in 3D world (X is down road, Z/Y is across road)
    const dx = ego.v * Math.cos(ego.psi) * dt;
    const dy = ego.v * Math.sin(ego.psi) * dt;
    ego.x += dx;
    ego.y += dy;
    state.distanceTraveled += Math.abs(ego.v) * dt;

    // Physical Pothole Impact & Suspension Dynamics (Accurate 4-Wheel Contact Model)
    const cosPsi = Math.cos(ego.psi);
    const sinPsi = Math.sin(ego.psi);
    const halfBase = 1.4;
    const halfTrack = 0.82;

    const egoWheelsPos = [
      { x: ego.x + halfBase * cosPsi - halfTrack * sinPsi, y: ego.y + halfBase * sinPsi + halfTrack * cosPsi }, // Front-Left
      { x: ego.x + halfBase * cosPsi + halfTrack * sinPsi, y: ego.y + halfBase * sinPsi - halfTrack * cosPsi }, // Front-Right
      { x: ego.x - halfBase * cosPsi - halfTrack * sinPsi, y: ego.y - halfBase * sinPsi + halfTrack * cosPsi }, // Rear-Left
      { x: ego.x - halfBase * cosPsi + halfTrack * sinPsi, y: ego.y - halfBase * sinPsi - halfTrack * cosPsi }  // Rear-Right
    ];

    for (const pot of state.potholes) {
      if (pot.hasHit) continue;
      let tireHit = false;
      for (const w of egoWheelsPos) {
        if (Math.hypot(w.x - pot.x, w.y - pot.y) < (pot.radius + 0.15)) {
          tireHit = true;
          break;
        }
      }

      if (tireHit) {
        pot.hasHit = true;
        if (ego.v <= 3.5) {
          // Crawl pace: vehicle successfully mitigated pothole impact
          ego.suspensionJolt = 0.055;
          ego.pitchJolt = -0.018;
          addEventLog('🕳️ Controlled pothole crossing: Crawl pace protected vehicle chassis & suspension.', 'text-amber-300 font-semibold');
        } else {
          // High speed strike: jarring impact
          ego.suspensionJolt = 0.24;
          ego.pitchJolt = -0.07;
          ego.v = Math.max(1.8, ego.v - 2.8);
          state.stats.potholeHits = (state.stats.potholeHits || 0) + 1;
          addEventLog('🕳️ POTHOLE STRIKE: High-speed rim impact! Suspension shock & rapid speed drop!', 'text-rose-400 font-bold');
        }
      }
    }

    // Update 3D Ego Mesh with Suspension Damped Oscillations
    egoGroup.position.set(ego.x, 0, ego.y);
    egoGroup.rotation.y = -ego.psi;

    if (ego.suspensionJolt > 0.001) {
      egoGroup.position.y = Math.sin(state.simTime * 32.0) * ego.suspensionJolt;
      egoGroup.rotation.z = Math.sin(state.simTime * 22.0) * (ego.pitchJolt || -0.05);
      ego.suspensionJolt *= 0.88;
      ego.pitchJolt = (ego.pitchJolt || 0.0) * 0.86;
    } else {
      egoGroup.position.y = 0;
      egoGroup.rotation.z = 0;
    }

    // Steer Front Wheels & Spin All Wheels
    const spinAngle = (ego.v * dt) / 0.38;
    egoWheels.forEach(w => {
      if (w.isFront) {
        w.pivot.rotation.y = -ego.delta;
      }
      w.wheelGroup.rotation.z -= spinAngle;
    });

    // Spin Roof LiDAR Puck
    if (egoLidarRotor) {
      egoLidarRotor.rotation.y += dt * 14.0;
    }

    // Dynamic Tail Light Bar (White reverse lights when backing up, intense red when braking)
    if (egoTailLightBar) {
      if (ego.v < -0.15 || ego.gear === 'R') {
        egoTailLightBar.material.emissiveIntensity = 3.2;
        egoTailLightBar.material.color.setHex(0xffffff);
        egoTailLightBar.material.emissive.setHex(0xffffff);
      } else if (ego.a < -0.35) {
        egoTailLightBar.material.emissiveIntensity = 2.5;
        egoTailLightBar.material.color.setHex(0xff0000);
        egoTailLightBar.material.emissive.setHex(0xff0000);
      } else {
        egoTailLightBar.material.emissiveIntensity = 0.9;
        egoTailLightBar.material.color.setHex(0xdc2626);
        egoTailLightBar.material.emissive.setHex(0xdc2626);
      }
    }

    // Dynamic Headlights (Clean, steady illumination without rapid strobing/blowing)
    if (egoHeadlights && egoHeadlights.length > 0) {
      let targetIntensity = 1.8;
      if (state.weather === 'night') targetIntensity = 4.5;
      else if (state.weather === 'rain' || state.weather === 'fog') targetIntensity = 3.6;
      else if (state.weather === 'sunset') targetIntensity = 2.6;
      egoHeadlights.forEach(hl => {
        hl.intensity = targetIntensity;
      });
    }

    // Dynamic Amber Turn Signals (Blink during lane change maneuvers)
    if (egoTurnSignals && egoTurnSignals.length > 0) {
      const isSignalingLeft = ego.targetD > ego.y + 0.35;
      const isSignalingRight = ego.targetD < ego.y - 0.35;
      const blinkOn = Math.sin(state.simTime * 9.0) > 0;
      egoTurnSignals.forEach(ts => {
        const shouldBlink = (ts.isRight && isSignalingRight) || (!ts.isRight && isSignalingLeft);
        ts.mesh.material.emissive.setHex(shouldBlink && blinkOn ? 0xf59e0b : 0x000000);
      });
    }

    // Continuous Dynamic Road Surface Pothole Spawning
    if (ego.x > state.lastPotholeSpawnX + 70.0) {
      state.lastPotholeSpawnX = ego.x;
      const aheadX = 65 + Math.random() * 40;
      const latChoices = [-2.0, -1.2, -0.4, 0.4, 1.2, 2.0];
      const potY = latChoices[Math.floor(Math.random() * latChoices.length)];
      const potR = 0.65 + Math.random() * 0.25;
      spawnPothole(aheadX, potY, potR);
    }

    // Recycle Potholes Falling Far Behind
    for (let i = state.potholes.length - 1; i >= 0; i--) {
      const pot = state.potholes[i];
      if (pot.x < ego.x - 50.0) {
        scene.remove(pot.mesh);
        state.potholes.splice(i, 1);
      }
    }

    // Continuous Dynamic Crossroad Lateral Traffic Spawning at Real Intersections Ahead
    if (ego.x > state.lastCrossroadSpawnX + 110.0) {
      state.lastCrossroadSpawnX = ego.x;
      const targetJunctionX = getUpcomingCrossroadX(28);
      spawnCrossroadActor(targetJunctionX);
    }

    // Update Dynamic Traffic Actors with mutual collision avoidance & solid obstacle consideration
    updateTrafficActors(dt);

    // Longitudinal & Lateral Physical Anti-Collision Clamp for Ego Vehicle (Zero Bumper Overlap / Sideswipe Prevention)
    for (const actor of state.trafficActors) {
      if (actor.isCrossroadActor) continue;
      const minLongGap = (ego.length + actor.length) * 0.5 + 0.85;
      const minLatGap = (ego.width + actor.width) * 0.5;
      const dx = actor.x - ego.x;
      const absDx = Math.abs(dx);
      const dy = actor.y - ego.y;
      const absDy = Math.abs(dy);

      // 1. Longitudinal clamp against lead vehicle ahead in same or overlapping path
      if (actor.speed >= 0 && dx > 0 && dx < minLongGap && absDy < minLatGap) {
        ego.x = actor.x - minLongGap;
        ego.v = Math.min(ego.v, Math.max(0.0, actor.speed));
        if (actor.speed <= 0.1) ego.v = 0.0;
      }

      // 2. Lateral sideswipe resolution: if vehicles overlap laterally and are alongside each other
      if (absDx < (ego.length + actor.length) * 0.46 && absDy < minLatGap) {
        const overlap = minLatGap - absDy + 0.05;
        if (ego.y >= actor.y) {
          ego.y += overlap * 0.6;
          actor.y -= overlap * 0.4;
        } else {
          ego.y -= overlap * 0.6;
          actor.y += overlap * 0.4;
        }
        actor.mesh.position.z = actor.y;
        egoGroup.position.z = ego.y;
        ego.v = Math.min(ego.v, Math.max(0.0, actor.speed));
      }

      // 3. Rear Physical Collision Clamp when reversing (Zero Bumper Overlap behind)
      if (ego.v < 0 && dx < 0 && absDx < minLongGap && absDy < minLatGap) {
        ego.x = actor.x + minLongGap;
        ego.v = 0.0;
        if (ego.a < 0) ego.a = 0.0;
      }
    }

    // Infinite Highway Recycling
    updateRoadSegments();

    // Dynamic Sunlight & Infinite Countryside Tracking (FOLLOWS EGO ALONG X!)
    dirLight.position.set(ego.x + 45, 80, 35);
    dirLight.target.position.set(ego.x + 10, 0, 0);
    dirLight.target.updateMatrixWorld();
    hemiLight.position.set(ego.x, 120, 0);
    terrainMesh.position.x = ego.x;
    if (rainParticleSystem.visible) {
      rainParticleSystem.position.x = ego.x;
    }

    // Update Visual Trajectory Ribbons
    updateTrajectoryVisuals();

    // Update Camera Perspective
    updateCamera();
  }

  // --- ADAPTIVE FRENET LATTICE PLANNER ---
  function runAdaptiveLatticePlanner(dt) {
    const ego = state.ego;
    state.stats.replans++;

    let minTTC = 99.9;
    let minClearance = 99.9;
    let closestHazardName = 'Roadway clear';

    // 1. Scan Traffic Actors
    for (const actor of state.trafficActors) {
      const dx = actor.x - ego.x;
      const dy = actor.y - ego.y;
      const dist = Math.hypot(dx, dy);

      if (dist < minClearance) {
        minClearance = dist;
        closestHazardName = `${actor.type.replace('_', ' ').toUpperCase()} (${dist.toFixed(1)}m)`;
      }

      // Closing speed & TTC
      const relSpeed = ego.v - actor.speed * Math.cos(ego.psi);
      if (dx > 0 && dx < 60 && relSpeed > 0.5) {
        const ttc = dx / relSpeed;
        if (ttc < minTTC) minTTC = ttc;
      }

      // Debounced Collision Detection
      if (Math.abs(dx) < (ego.length + actor.length) * 0.44 && Math.abs(dy) < (ego.width + actor.width) * 0.45) {
        if (!actor.hasCollided) {
          actor.hasCollided = true;
          state.stats.collisions++;
          addEventLog(`💥 COLLISION DETECTED with ${actor.type.toUpperCase()}!`, 'text-red-400 font-bold');
        }
      } else if (dist < 1.8) {
        state.stats.nearMisses++;
      } else if (dist > 5.0) {
        actor.hasCollided = false;
      }
    }

    // 2. Scan Potholes Ahead for Hazard Status & Active Braking
    let closestPothole = null;
    let minPotholeDist = 999;
    for (const pot of state.potholes) {
      const dx = pot.x - ego.x;
      const dy = pot.y - ego.y;
      if (dx > 0 && dx < 35.0) {
        if (Math.abs(dy) < (pot.radius + 1.25)) {
          if (dx < minPotholeDist) {
            minPotholeDist = dx;
            closestPothole = pot;
          }
        }
      }
    }

    if (closestPothole && minPotholeDist < minClearance) {
      minClearance = minPotholeDist;
      closestHazardName = `POTHOLE (${minPotholeDist.toFixed(1)}m)`;
    }

    ego.minTTC = minTTC;
    ego.minClearance = minClearance;
    ego.closestHazard = closestHazardName;

    // Multi-Factor Risk Assessment
    if (minTTC < 1.8 || minClearance < 2.5) {
      ego.riskLevel = 'CRITICAL';
    } else if (minTTC < 3.2 || minClearance < 5.0) {
      ego.riskLevel = 'HIGH';
    } else if (minTTC < 5.5 || minClearance < 9.0) {
      ego.riskLevel = 'MEDIUM';
    } else {
      ego.riskLevel = 'LOW';
    }

    // 3. Multi-Candidate Frenet Lattice Path Optimization (11 High-Resolution Paths)
    const candidateOffsets = [-2.8, -2.2, -1.65, -1.1, -0.55, 0.0, 0.55, 1.1, 1.65, 2.2, 2.8];
    let bestOffset = 0.0;
    let minCost = Infinity;

    // Scan for lead vehicles in current lane to check if overtake is beneficial
    let leadInCurrentLane = null;
    let leadInCurrentDist = 999;
    for (const actor of state.trafficActors) {
      if (actor.isCrossroadActor || actor.speed < 0) continue;
      const dx = actor.x - ego.x;
      if (dx > 0 && dx < 36.0 && Math.abs(actor.y - ego.y) < 1.35) {
        if (dx < leadInCurrentDist) {
          leadInCurrentDist = dx;
          leadInCurrentLane = actor;
        }
      }
    }

    candidateOffsets.forEach(candD => {
      let cost = 0.0;

      // Road boundary compliance (safe drivable half-width is ~3.2m)
      const absCand = Math.abs(candD);
      if (absCand > 2.8) {
        cost += (absCand - 2.8) * 1200.0;
      }
      if (absCand > 3.3) {
        cost += 25000.0; // Strictly outside road boundaries!
      }

      // Proximity & Collision Risk across all Traffic Actors
      for (const actor of state.trafficActors) {
        if (actor.isCrossroadActor) {
          const cDx = actor.x - ego.x;
          if (cDx > -2.0 && cDx < 38.0 && Math.abs(actor.y) < 5.5) {
            cost += 15000.0 / (1.0 + (cDx / 6.0) ** 2);
          }
          continue;
        }

        const latDist = Math.abs(candD - actor.y);
        const longDist = actor.x - ego.x;
        const safeLatGap = (ego.width + actor.width) * 0.5 + 0.45;

        // 1. SIDESWIPE PREVENTION: Actor is alongside ego or in immediate blind spot
        // Never turn into or cut across a vehicle beside us!
        if (longDist > -8.0 && longDist < 8.0) {
          if (latDist < safeLatGap) {
            cost += 48000.0 / Math.max(0.08, latDist);
          }
          const curLatDist = Math.abs(ego.y - actor.y);
          if (curLatDist < safeLatGap + 0.35 && Math.abs(candD - actor.y) < curLatDist) {
            cost += 40000.0;
          }
        }

        // 2. ONCOMING TRAFFIC: Strict absolute no-go
        if (actor.speed < 0) {
          if (longDist > -6.0 && longDist < 70.0 && latDist < safeLatGap + 0.35) {
            cost += 55000.0 / (1.0 + (Math.max(0, longDist) / 10.0) ** 2) + 25000.0;
          }
          continue;
        }

        // 3. FORWARD TRAFFIC (Lead vehicles in same or target lanes)
        if (longDist >= 0 && longDist < 55.0) {
          if (latDist < safeLatGap) {
            // Vehicle occupies this candidate lane ahead
            cost += 32000.0 / (1.0 + (longDist / 6.0) ** 2);

            // Strict Overtake Headway Check: Never jump into an overtaking lane without a safe opening!
            // If candidate offset represents a lane change (|candD - ego.y| > 0.6), require at least 24m gap!
            if (Math.abs(candD - ego.y) > 0.6 && longDist < 24.0) {
              cost += (24.0 - longDist) * 2200.0;
            }
          }
        }

        // 4. SWEPT TRANSITION CORRIDOR (Eliminates sideswiping during lane change maneuvers)
        const sweepMin = Math.min(ego.y, candD) - safeLatGap * 0.82;
        const sweepMax = Math.max(ego.y, candD) + safeLatGap * 0.82;
        if (actor.y >= sweepMin && actor.y <= sweepMax && longDist > -5.0 && longDist < 22.0) {
          cost += 36000.0 / (1.0 + (Math.max(0, longDist) / 5.0) ** 2);
        }

        // 5. SAFE RETURN AFTER OVERTAKE: Ensure ego doesn't cut back in front of overtaken vehicle too sharply
        if (longDist < 0 && longDist > -12.0 && latDist < safeLatGap) {
          cost += 12000.0 / (1.0 + (Math.abs(longDist) / 3.5) ** 2);
        }

        // 6. Cattle & pedestrians
        if (actor.type === 'cattle') {
          const normDist = (longDist / 10.0) ** 2 + (latDist / safeLatGap) ** 2;
          cost += 16000.0 / Math.max(0.12, normDist);
        }
      }

      // OVERTAKE INCENTIVE (Only if current lead is slow AND target lane is genuinely clear!):
      if (leadInCurrentLane && leadInCurrentDist < 28.0 && leadInCurrentLane.speed < 10.5) {
        if (Math.abs(candD - leadInCurrentLane.y) > (ego.width + leadInCurrentLane.width) * 0.5 + 0.65 && cost < 1500.0) {
          // Clean open lane available: reward safe overtake
          cost -= Math.max(0, 11.5 - leadInCurrentLane.speed) * 140.0;
        }
      }

      // POTHOLE AVOIDANCE (Surface defect, NOT an impenetrable wall!):
      // - Only avoid potholes within natural lookahead distance (dx between 2.0m and 24.0m).
      // - If dx >= 24.0m: IGNORE! (Zero cost: prevents premature, illogical lane changes 50m away!).
      // - Substantial avoidance cost (1500 - 3800): vehicle steers around pothole IF the adjacent lane is safe,
      //   BUT if adjacent lane has another vehicle (cost > 30,000) or boundary, ego stays in lane and crawls across!
      for (const pot of state.potholes) {
        const dx = pot.x - ego.x;
        if (dx > 2.0 && dx < 24.0) {
          const dLeft = Math.abs((candD + 0.82) - pot.y);
          const dRight = Math.abs((candD - 0.82) - pot.y);
          const minTireDist = Math.min(dLeft, dRight);
          const underbodyDist = Math.abs(candD - pot.y);

          const hitTire = minTireDist < (pot.radius + 0.25);
          const hitUnderbody = underbodyDist < (pot.radius + 0.65);

          if (hitTire || hitUnderbody) {
            const distFactor = (1.0 - (dx - 2.0) / 22.0); // 0 at 24m, 1.0 at 2m (smooth, natural avoidance)
            const severity = hitTire ? 1.0 : 0.6;
            cost += 3800.0 * distFactor * severity;
          }
        }
      }

      // Lane preference: slight bias toward nominal centerline/right lane
      cost += Math.abs(candD - 0.0) * 4.0;

      // Smoothness cost to prevent erratic high-frequency weaving
      cost += ((candD - ego.targetD) ** 2) * 35.0;

      if (cost < minCost) {
        minCost = cost;
        bestOffset = candD;
      }
    });

    ego.targetD = bestOffset;

    // 4. Longitudinal Speed Planning & Active Braking
    let targetSpeed = 12.5; // Nominal cruise: ~45 km/h
    let activeManeuver = 'Cruise';
    let plannerState = 'NORMAL_DRIVING';

    let forwardObstacleDist = 999;
    let forwardObstacleType = null;
    let forwardObstacleSpeed = 12.5;

    // Check for Crossroad Traffic cutting across intersection ahead
    let crossroadActor = null;
    let crossroadDx = 999;
    for (const actor of state.trafficActors) {
      if (actor.isCrossroadActor || Math.abs(actor.vy || 0) > 1.5) {
        const dx = actor.x - ego.x;
        // Detect crossroad vehicle from well ahead (up to 48m) as it approaches or crosses the roadway
        if (dx > -3.5 && dx < 48.0) {
          const isEnteringOrCrossing = Math.abs(actor.y) < 5.8 ||
            (actor.y > 0 && (actor.vy || 0) < -0.1) ||
            (actor.y < 0 && (actor.vy || 0) > 0.1);
          if (isEnteringOrCrossing && Math.abs(actor.y) < 18.0) {
            if (dx < crossroadDx) {
              crossroadDx = dx;
              crossroadActor = actor;
            }
          }
        }
      }
    }

    // Check for Cut-In vehicles closing laterally into ego's lane
    let cutInActor = null;
    let cutInDx = 999;
    let cutInSpeed = 0.0;
    for (const actor of state.trafficActors) {
      if (actor.isCrossroadActor) continue;
      const dx = actor.x - ego.x;
      const latGap = Math.abs(actor.y - ego.y);
      const isClosingIn = (actor.y > ego.y && (actor.vy || 0) < -0.35) || (actor.y < ego.y && (actor.vy || 0) > 0.35);
      if (dx > 0 && dx < 28.0 && (latGap < 1.75 || isClosingIn)) {
        if (dx < cutInDx) {
          cutInDx = dx;
          cutInActor = actor;
          cutInSpeed = actor.speed;
        }
      }
    }

    // Scan forward obstacles in current lane, target lane, and swept transition corridor (Eliminates dead zones during lane changes!)
    for (const actor of state.trafficActors) {
      if (actor.isCrossroadActor) continue;
      const dx = actor.x - ego.x;
      const safeLatDist = (ego.width + actor.width) * 0.5 + 0.35;
      const latDistCurrent = Math.abs(actor.y - ego.y);
      const latDistTarget = Math.abs(actor.y - ego.targetD);
      const inSweepCorridor = (actor.y >= Math.min(ego.y, ego.targetD) - safeLatDist && actor.y <= Math.max(ego.y, ego.targetD) + safeLatDist);

      if (dx > 0 && dx < 55.0 && (latDistCurrent < safeLatDist || latDistTarget < safeLatDist || inSweepCorridor)) {
        if (dx < forwardObstacleDist) {
          forwardObstacleDist = dx;
          forwardObstacleType = actor.type;
          forwardObstacleSpeed = actor.speed;
        }
      }
    }

    // Pothole forward crawl check: If pothole is in upcoming path (< 22m), engage active braking to cross at safe crawl pace!
    for (const pot of state.potholes) {
      const dx = pot.x - ego.x;
      if (dx > 0 && dx < 22.0) {
        const curDistToTire = Math.min(Math.abs((ego.y + 0.82) - pot.y), Math.abs((ego.y - 0.82) - pot.y));
        const curUnderbody = Math.abs(ego.y - pot.y) < (pot.radius + 0.65);
        const willHitCurrent = curDistToTire < (pot.radius + 0.25) || curUnderbody;

        const targetDistToTire = Math.min(Math.abs((ego.targetD + 0.82) - pot.y), Math.abs((ego.targetD - 0.82) - pot.y));
        const targetUnderbody = Math.abs(ego.targetD - pot.y) < (pot.radius + 0.65);
        const willHitTarget = targetDistToTire < (pot.radius + 0.25) || targetUnderbody;

        if (willHitCurrent || willHitTarget) {
          // Pothole crossing in progress or unavoidable: apply brakes to cross at safe suspension crawl pace!
          if (dx < forwardObstacleDist) {
            forwardObstacleDist = dx;
            forwardObstacleType = 'pothole';
            forwardObstacleSpeed = 2.2; // Safe crawl pace
          }
        }
      }
    }

    if (crossroadActor) {
      // EGO YIELDS TO CROSS TRAFFIC: Ego vehicle stops before intersection to let cross-traffic pass
      closestHazardName = `CROSS TRAFFIC (${crossroadDx.toFixed(1)}m)`;
      if (crossroadDx < 8.0) {
        // Ego stops and holds completely at the stop line before the intersection!
        targetSpeed = 0.0;
        activeManeuver = 'Yielding to Crossroad Traffic (Stop Line)';
        plannerState = 'CROSSROAD_STOP';
      } else if (crossroadDx < 20.0) {
        // Progressive braking to stop smoothly before reaching intersection
        targetSpeed = Math.max(0.0, Math.min(2.5, (crossroadDx - 7.0) * 0.45));
        activeManeuver = 'Braking for Crossroad Traffic';
        plannerState = 'CROSSROAD_YIELD';
      } else {
        // Approaching intersection: prudent deceleration
        targetSpeed = Math.min(targetSpeed, 4.5);
        activeManeuver = 'Approaching Intersection';
        plannerState = 'CROSSROAD_APPROACH';
      }

      if (!crossroadActor.hasAlerted) {
        crossroadActor.hasAlerted = true;
        const originSide = (crossroadActor.originY || crossroadActor.y) > 0 ? 'LEFT' : 'RIGHT';
        addEventLog(`🚦 JUNCTION YIELD: Cross-traffic approaching from ${originSide} (${crossroadDx.toFixed(1)}m). Ego vehicle stopping to yield right-of-way.`, 'text-cyan-300 font-bold');
      }

    } else if (cutInActor && cutInDx < 22.0) {
      // Cut-in vehicle evasion & braking
      closestHazardName = `CUT-IN VEHICLE (${cutInDx.toFixed(1)}m)`;
      if (cutInDx < 5.5) {
        targetSpeed = 0.0;
        activeManeuver = 'Emergency Cut-In Brake';
        plannerState = 'EMERGENCY_STOP';
      } else if (cutInDx < 12.0) {
        targetSpeed = Math.max(0.0, Math.min(cutInSpeed * 0.7, (cutInDx - 4.5) * 1.2));
        activeManeuver = 'Yielding to Cut-In Vehicle';
        plannerState = 'CUT_IN_EVASION';
      } else {
        targetSpeed = Math.max(0.0, Math.min(cutInSpeed * 0.88, (cutInDx - 4.5) * 0.9));
        activeManeuver = 'Managing Gap to Cut-In';
        plannerState = 'CUT_IN_EVASION';
      }
    } else if (forwardObstacleType === 'pothole') {
      // Controlled Pothole Crossing: Surface defect is NOT a wall!
      // Actively apply brakes to cross at safe suspension crawl pace (~2.2 m/s = 8 km/h) and roll across cleanly!
      if (forwardObstacleDist < 6.0) {
        targetSpeed = 2.2; // Safe crawl pace: rolls across safely without stopping or reversing
        activeManeuver = 'Controlled Pothole Crawl [2.2 m/s]';
        plannerState = 'POTHOLE_CROSSING';
      } else {
        // Smooth, progressive deceleration down to crawl pace
        targetSpeed = Math.max(2.2, Math.min(6.5, 2.2 + (forwardObstacleDist - 5.0) * 0.35));
        activeManeuver = 'Braking for Pothole Crossing';
        plannerState = 'POTHOLE_APPROACH';
      }
    } else if (forwardObstacleType === 'cattle' || forwardObstacleSpeed < 1.0) {
      // Fixed or slow-moving obstacle (cow, stopped vehicle)
      if (forwardObstacleDist < 9.0) {
        targetSpeed = 0.0;
        activeManeuver = 'Emergency Stop (Yielding to Hazard)';
        plannerState = 'EMERGENCY_STOP';
      } else if (forwardObstacleDist < 18.0) {
        targetSpeed = 3.5;
        activeManeuver = 'Yielding to Hazard Ahead';
        plannerState = 'YIELD_CORRIDOR';
      } else if (forwardObstacleDist < 30.0) {
        targetSpeed = 6.5;
        activeManeuver = 'Cautious Approach';
        plannerState = 'ADAPTIVE_FOLLOW';
      }
    } else if (forwardObstacleType) {
      // Moving vehicle ahead: maintain comfortable safe headway gap with STRICT STANDSTILL BUFFER (NO COLLISIONS)
      const safeHeadwayGap = 6.5 + ego.v * 1.8;
      const closingSpeed = Math.max(0, ego.v - forwardObstacleSpeed);

      if (forwardObstacleDist < 5.0) {
        // MANDATORY STANDSTILL BUFFER: Must stop completely at least 5.0m away! Zero contact.
        targetSpeed = 0.0;
        activeManeuver = 'Standstill Hold (Safe Buffer)';
        plannerState = 'STANDSTILL_HOLD';
      } else if (forwardObstacleDist < 12.0) {
        // Decelerate decisively to match or drop below lead speed to open distance
        const decelRatio = Math.max(0.0, (forwardObstacleDist - 5.0) / 7.0);
        targetSpeed = Math.max(0.0, Math.min(forwardObstacleSpeed * 0.7, decelRatio * forwardObstacleSpeed));
        activeManeuver = 'Braking behind Lead Vehicle';
        plannerState = 'DECEL_APPROACH';
      } else if (forwardObstacleDist < safeHeadwayGap || closingSpeed > 1.2) {
        // Match lead vehicle speed with safe buffer
        targetSpeed = Math.max(0.0, Math.min(forwardObstacleSpeed, forwardObstacleSpeed * 0.92));
        activeManeuver = `Car-Following [${(forwardObstacleSpeed * 3.6).toFixed(0)} km/h]`;
        plannerState = 'ADAPTIVE_CRUISE';
      } else {
        targetSpeed = Math.min(12.5, forwardObstacleSpeed + (forwardObstacleDist - safeHeadwayGap) * 0.4);
        activeManeuver = 'Cruising behind Traffic';
        plannerState = 'NORMAL_DRIVING';
      }
    } else {
      // Path is clear
      targetSpeed = 12.5;
      if (Math.abs(ego.targetD - ego.y) > 0.4) {
        activeManeuver = ego.targetD > ego.y ? 'Overtake Left' : 'Maneuver Right';
        plannerState = 'EVASIVE_OVERTAKE';
      } else {
        activeManeuver = 'Cruise';
        plannerState = 'NORMAL_DRIVING';
      }
    }

    // Safety check: Scan traffic behind the vehicle to ensure reversing corridor is clear
    let rearClear = true;
    let minRearDist = 999;
    for (const actor of state.trafficActors) {
      const rearDist = (ego.x - actor.x);
      if (rearDist > 0 && rearDist < 25.0 && Math.abs(actor.y - ego.y) < 1.65) {
        if (rearDist < minRearDist) minRearDist = rearDist;
        if (rearDist < 4.5) {
          rearClear = false;
        }
      }
    }

    // 5. Autonomous Reversing Decision Engine (Deadlock Escape & Bottleneck Yielding)
    // Solid obstacles only! Potholes are surface defects, NOT walls, and NEVER cause deadlocks or reversing!
    const isSolidObstacle = forwardObstacleType && forwardObstacleType !== 'pothole';
    const isForwardTrapped = isSolidObstacle && (forwardObstacleDist < 4.8 || minClearance < 2.6);
    const isStoppedOrCrawling = Math.abs(ego.v) < 0.6;
    const isForwardCorridorBlocked = minCost > 4500 || forwardObstacleDist < 3.8;

    if (isForwardTrapped && isStoppedOrCrawling && isForwardCorridorBlocked && !crossroadActor) {
      ego.deadlockTimer = (ego.deadlockTimer || 0) + dt;
      if (ego.deadlockTimer > 1.8 && rearClear && !ego.isReversing) {
        ego.isReversing = true;
        ego.reverseTimer = 3.5; // Back up for ~3.5s to create ~7m opening
        ego.reverseTargetD = ego.y > 0 ? 2.6 : -2.6; // Angle rear toward road shoulder
        addEventLog('⏪ AUTONOMOUS REVERSE: Forward corridor deadlocked! Backing up to yield & create clearance.', 'text-amber-300 font-bold');
      }
    } else if (forwardObstacleDist > 7.0 && minClearance > 4.5) {
      ego.deadlockTimer = 0.0;
    }

    // Active Autonomous Reverse Execution
    if (ego.isReversing) {
      if (!rearClear) {
        // Vehicle approached from rear: halt reverse immediately to avoid rear collision
        ego.isReversing = false;
        ego.reverseTimer = 0.0;
        targetSpeed = 0.0;
        activeManeuver = 'Reverse Halted (Rear Proximity)';
        plannerState = 'STANDSTILL_HOLD';
      } else {
        ego.reverseTimer -= dt;
        targetSpeed = -2.2; // Steady, controlled reverse speed
        ego.gear = 'R';
        activeManeuver = 'Autonomous Reverse Yield [R]';
        plannerState = 'REVERSE_DEADLOCK_ESCAPE';

        // Steer rear toward road shoulder while backing up
        const latErr = ego.y - ego.reverseTargetD;
        ego.targetDelta = Math.max(-ego.maxSteer * 0.5, Math.min(ego.maxSteer * 0.5, latErr * 0.45));

        // Exit reverse when timer expires or sufficient clearance created: Hold safe standstill, do NOT lurch forward!
        if (ego.reverseTimer <= 0 || (forwardObstacleDist > 9.0 && minClearance > 5.5)) {
          ego.isReversing = false;
          ego.reverseTimer = 0.0;
          ego.deadlockTimer = 0.0;
          targetSpeed = 0.0; // Standstill hold: wait until lead vehicle moves or corridor is genuinely clear
          ego.gear = 'D';
          activeManeuver = 'Holding Standstill Gap';
          plannerState = 'STANDSTILL_HOLD';
          addEventLog('✅ Autonomous reverse complete: Safe corridor restored. Holding safe gap.', 'text-emerald-400 font-semibold');
        }
      }
    } else if (isForwardTrapped && isStoppedOrCrawling && rearClear && forwardObstacleDist < 2.5 && !crossroadActor) {
      // Controlled reverse buffer if solid obstacle encroaches dangerously close
      ego.isReversing = true;
      ego.reverseTimer = 2.2;
      ego.reverseTargetD = ego.y > 0 ? 2.5 : -2.5;
    }

    ego.targetV = targetSpeed;
    ego.activeManeuver = activeManeuver;
    ego.plannerState = plannerState;

    // Stanley Lateral Tracking Controller (Active during forward drive)
    if (!ego.isReversing && targetSpeed >= 0) {
      const e_lat = ego.y - ego.targetD;
      const k_stanley = 0.95; // Crisp, responsive lateral tracking
      const deltaStanley = -ego.psi - Math.atan2(k_stanley * e_lat, Math.max(1.5, Math.abs(ego.v)));
      // Speed-scaled dynamic steer saturation to prevent violent swerving and overshoot during overtakes
      const speedScaledSteerLimit = Math.max(0.18, Math.min(ego.maxSteer, 2.4 / Math.max(2.5, Math.abs(ego.v))));
      ego.targetDelta = Math.max(-speedScaledSteerLimit, Math.min(speedScaledSteerLimit, deltaStanley));
    }

    // Longitudinal Smooth Acceleration Controller with Rock-Solid Standstill Hold (No Reverse Drifting or Oscillation)
    if (targetSpeed === 0.0) {
      if (Math.abs(ego.v) < 0.12) {
        // PRECISE ZERO STANDSTILL LOCK: Prevents negative velocity creep and back-and-forth oscillation
        ego.v = 0.0;
        ego.a = 0.0;
        ego.gear = 'D';
      } else if (ego.v > 0.12) {
        // Smoothly and firmly brake forward motion to stop
        let brakeStrength = Math.max(2.0, ego.v * 2.2);
        if (forwardObstacleDist < 5.5 || crossroadDx < 18.0) {
          brakeStrength = Math.max(3.8, ego.v * 2.8);
        }
        const desiredA = Math.max(-ego.maxDecel, -brakeStrength);
        ego.a += (desiredA - ego.a) * Math.min(1.0, dt * 6.0);
      } else {
        // Braking from reverse motion: apply positive deceleration force to bring v up to 0
        const desiredA = Math.min(ego.maxAccel, Math.max(2.2, -ego.v * 2.8));
        ego.a += (desiredA - ego.a) * Math.min(1.0, dt * 6.0);
      }
    } else if (targetSpeed < 0) {
      // Reverse speed control
      const vErr = targetSpeed - ego.v;
      const desiredA = Math.max(-2.5, Math.min(2.5, vErr * 1.8));
      ego.a += (desiredA - ego.a) * Math.min(1.0, dt * 5.0);
    } else {
      // Forward drive control
      const vErr = targetSpeed - ego.v;
      let desiredA = vErr * 1.8;
      if (vErr < -1.5) desiredA = Math.min(desiredA, -3.5); // Stronger braking when over speed
      if (forwardObstacleDist < 8.5 && forwardObstacleType && forwardObstacleType !== 'pothole') {
        desiredA = Math.min(desiredA, -5.2); // Responsive braking when closing in behind lead vehicle
      }
      desiredA = Math.max(-ego.maxDecel, Math.min(ego.maxAccel, desiredA));
      ego.a += (desiredA - ego.a) * Math.min(1.0, dt * 6.0);
    }
  }

  // --- MANUAL DRIVING CONTROLLER (WASD / ARROW KEYS WITH REVERSE GEAR) ---
  function runManualController(dt) {
    const ego = state.ego;

    if (state.keys.forward) {
      if (ego.v < -0.25) {
        // Reversing brake: firm deceleration to stop backward motion
        ego.a = 6.0;
        ego.activeManeuver = 'Reverse Braking';
        ego.plannerState = 'MANUAL_BRAKE';
      } else {
        ego.a = 3.2;
        ego.gear = 'D';
        ego.activeManeuver = 'Manual Forward [D]';
        ego.plannerState = 'MANUAL_DRIVE';
      }
    } else if (state.keys.backward) {
      if (ego.v > 0.25) {
        // Forward brake: decelerate down to zero
        ego.a = -6.5;
        ego.activeManeuver = 'Manual Braking';
        ego.plannerState = 'MANUAL_BRAKE';
      } else {
        // Reverse throttle: accelerate backward up to -4.5 m/s!
        ego.a = -2.8;
        ego.gear = 'R';
        ego.activeManeuver = 'Manual Reverse [R]';
        ego.plannerState = 'MANUAL_REVERSE';
      }
    } else {
      // Natural rolling resistance friction to bring vehicle to 0
      if (Math.abs(ego.v) < 0.18) {
        ego.v = 0.0;
        ego.a = 0.0;
        ego.gear = 'P';
        ego.activeManeuver = 'Standstill [P]';
        ego.plannerState = 'STANDSTILL';
      } else {
        ego.a = -Math.sign(ego.v) * 1.5;
        ego.activeManeuver = ego.v > 0 ? 'Coasting Forward' : 'Coasting Reverse [R]';
      }
    }

    if (state.keys.left) {
      ego.targetDelta = ego.maxSteer * 0.7;
    } else if (state.keys.right) {
      ego.targetDelta = -ego.maxSteer * 0.7;
    } else {
      ego.targetDelta = 0.0;
    }
  }

  // --- DEDICATED REVERSE MODE CONTROLLER (NO FORWARD SURGES OR OSCILLATION WITH FRONT OBSTACLES) ---
  function runReverseController(dt) {
    const ego = state.ego;
    ego.gear = 'R';

    // Scan traffic behind the vehicle to ensure reversing corridor is safe
    let rearClear = true;
    let minRearDist = 999;
    for (const actor of state.trafficActors) {
      const rearDist = (ego.x - actor.x);
      if (rearDist > 0 && rearDist < 28.0 && Math.abs(actor.y - ego.y) < 1.75) {
        if (rearDist < minRearDist) minRearDist = rearDist;
        if (rearDist < 4.5) {
          rearClear = false;
        }
      }
    }

    if (!rearClear) {
      // Rear obstacle detected: halt reverse safely to prevent rear collision
      ego.targetV = 0.0;
      ego.activeManeuver = `Reverse Halted (Rear Proximity ${minRearDist.toFixed(1)}m)`;
      ego.plannerState = 'STANDSTILL_HOLD';
    } else {
      // Steady, controlled reverse speed (-2.2 m/s = ~8 km/h)
      // Front obstacles do NOT trigger forward drive in reverse mode!
      ego.targetV = -2.2;
      ego.activeManeuver = 'Reverse Driving [R]';
      ego.plannerState = 'REVERSE_MODE';

      // Keep car parallel to lane / slight bias toward road shoulder
      const targetShoulderY = ego.y > 0 ? 2.5 : -2.5;
      const latErr = ego.y - targetShoulderY;
      ego.targetDelta = Math.max(-ego.maxSteer * 0.45, Math.min(ego.maxSteer * 0.45, latErr * 0.35));
    }

    // Longitudinal Controller in Reverse Mode
    if (ego.targetV === 0.0) {
      if (Math.abs(ego.v) < 0.12) {
        ego.v = 0.0;
        ego.a = 0.0;
      } else if (ego.v < -0.12) {
        // Firm braking of backward motion to stop
        ego.a = Math.min(ego.maxAccel, Math.max(2.0, -ego.v * 3.0));
      } else {
        ego.a = Math.max(-ego.maxDecel, Math.min(-2.0, -ego.v * 3.0));
      }
    } else {
      // Smooth tracking of reverse target velocity (-2.2 m/s)
      const vErr = ego.targetV - ego.v;
      const desiredA = Math.max(-2.5, Math.min(2.5, vErr * 1.8));
      ego.a += (desiredA - ego.a) * Math.min(1.0, dt * 5.0);
    }
  }

  // --- UPDATE DYNAMIC TRAFFIC ACTORS (MUTUAL AVOIDANCE & SOLID OBSTACLE AWARENESS) ---
  function updateTrafficActors(dt) {
    const ego = state.ego;

    // 1. Behavior and Trajectory Update for Each Actor
    for (let i = state.trafficActors.length - 1; i >= 0; i--) {
      const actor = state.trafficActors[i];

      if (actor.isCrossroadActor) {
        // Lateral crossroad traffic in designated cross street lane has right-of-way
        let leadCrossActorInFront = false;

        // 1. Check if another crossroad vehicle is ahead in the same lane
        for (let j = 0; j < state.trafficActors.length; j++) {
          if (i === j) continue;
          const other = state.trafficActors[j];
          if (other.isCrossroadActor) {
            const yAhead = (other.y - actor.y) * Math.sign(actor.nominalVy);
            const xGap = Math.abs(other.x - actor.x);
            if (yAhead > 0 && yAhead < 6.5 && xGap < 2.5) {
              leadCrossActorInFront = true;
              break;
            }
          }
        }

        if (leadCrossActorInFront) {
          actor.targetVy = 0.0; // Wait behind leading crossroad vehicle
        } else {
          // Crossroad traffic has priority: NEVER yields or stops for ego!
          actor.targetVy = actor.nominalVy;
        }

        // Smooth speed adjustment
        const vyDiff = actor.targetVy - actor.vy;
        actor.vy += Math.sign(vyDiff) * Math.min(Math.abs(vyDiff), 5.5 * dt);

        actor.y += actor.vy * dt;
        actor.mesh.position.set(actor.x, 0, actor.y);
        actor.mesh.rotation.y = -actor.heading;

        // Spin wheels for moving crossroad vehicles
        const spinAngle = (Math.abs(actor.vy) * dt) / 0.35;
        actor.mesh.traverse(child => {
          if (child.isMesh && child.geometry && child.geometry.type === 'CylinderGeometry' && child.scale.y < 0.8) {
            child.rotation.x += spinAngle;
          }
        });

        // Recycle when cleared to the opposite side
        if (Math.abs(actor.y) > 26.0) {
          scene.remove(actor.mesh);
          state.trafficActors.splice(i, 1);
        }
        continue;

      } else if (actor.type === 'cattle') {
        actor.behaviorTimer -= dt;
        const distToEgo = Math.hypot(actor.x - ego.x, actor.y - ego.y);

        // 1. VEHICLE CONTACT DETECTION (EGO OR ANY TRAFFIC VEHICLE):
        // If an animal hits or is contacted by any vehicle, it IMMEDIATELY reverses its direction of movement!
        let hitVehicle = null;

        // Contact with ego vehicle
        const egoDx = Math.abs(actor.x - ego.x);
        const egoDy = Math.abs(actor.y - ego.y);
        const minEgoColX = (ego.length + actor.length) * 0.48;
        const minEgoColY = (ego.width + actor.width) * 0.48;
        if (egoDx < minEgoColX && egoDy < minEgoColY) {
          hitVehicle = { x: ego.x, y: ego.y, speed: ego.v };
        }

        // Contact with other traffic actors
        if (!hitVehicle) {
          for (let j = 0; j < state.trafficActors.length; j++) {
            if (i === j) continue;
            const other = state.trafficActors[j];
            if (other.type === 'cattle') continue;
            const oDx = Math.abs(actor.x - other.x);
            const oDy = Math.abs(actor.y - other.y);
            const minColX = (other.length + actor.length) * 0.48;
            const minColY = (other.width + actor.width) * 0.48;
            if (oDx < minColX && oDy < minColY) {
              hitVehicle = { x: other.x, y: other.y, speed: other.speed || other.vy || 0 };
              break;
            }
          }
        }

        // Debounced collision reaction (at least 1.5s between direction reversals)
        const now = state.simTime;
        if (hitVehicle && (!actor.lastHitTime || now - actor.lastHitTime > 1.5)) {
          actor.lastHitTime = now;

          // REVERSE DIRECTION OF MOVEMENT UPON VEHICLE CONTACT:
          // If was moving towards left (vy > 0), turn and scurry towards right!
          // If was moving towards right (vy < 0), turn and scurry towards left!
          // If standing or near zero velocity, retreat away from vehicle center!
          let fleeToLeft = false;
          if (actor.vy > 0.08) {
            fleeToLeft = false; // Reverse towards right
          } else if (actor.vy < -0.08) {
            fleeToLeft = true; // Reverse towards left
          } else {
            fleeToLeft = actor.y >= hitVehicle.y;
          }

          if (fleeToLeft) {
            actor.behaviorState = 'crossing_to_left';
            actor.vy = actor.walkSpeed * 1.35;
          } else {
            actor.behaviorState = 'crossing_to_right';
            actor.vy = -actor.walkSpeed * 1.35;
          }
          actor.vx = 0.08;
          actor.behaviorTimer = 6.0 + Math.random() * 3.0; // Lock this retreat direction for 6-9s!
          actor.heading = actor.vy > 0 ? Math.PI / 2 : -Math.PI / 2;

          addEventLog(`🐄 Animal contacted by vehicle: Reversed direction of movement towards ${fleeToLeft ? 'left' : 'right'} shoulder!`, 'text-yellow-300 font-bold');
        }

        // 2. STEADY, DELIBERATE MOVEMENT (NO RAPID PING-PONG LEFT/RIGHT):
        if (actor.behaviorState === 'crossing_to_left') {
          // Steadily crosses towards left shoulder
          actor.vy = actor.walkSpeed;
          actor.vx = 0.04;
          if (actor.y > 3.6) {
            // Reached left shoulder: transition to calm shoulder grazing
            actor.behaviorState = 'grazing_shoulder';
            actor.behaviorTimer = 8.0 + Math.random() * 6.0;
            actor.vy = 0.0;
            actor.vx = 0.03;
          }
        } else if (actor.behaviorState === 'crossing_to_right') {
          // Steadily crosses towards right shoulder
          actor.vy = -actor.walkSpeed;
          actor.vx = 0.04;
          if (actor.y < -3.6) {
            // Reached right shoulder: transition to calm shoulder grazing
            actor.behaviorState = 'grazing_shoulder';
            actor.behaviorTimer = 8.0 + Math.random() * 6.0;
            actor.vy = 0.0;
            actor.vx = 0.03;
          }
        } else if (actor.behaviorState === 'standing_calm') {
          // Calm pause when vehicle yields in front
          actor.vx = 0.0;
          actor.vy = 0.0;
          if (actor.behaviorTimer <= 0) {
            actor.behaviorState = actor.resumeDirection || (actor.y >= 0 ? 'crossing_to_left' : 'crossing_to_right');
            actor.behaviorTimer = 8.0;
          }
        } else if (actor.behaviorState === 'grazing_shoulder') {
          // Calm grazing on shoulder grass without twitching
          actor.vx = 0.02;
          actor.vy = 0.0;
          if (actor.behaviorTimer <= 0) {
            actor.behaviorTimer = 10.0 + Math.random() * 6.0;
            if (Math.random() > 0.45) {
              actor.behaviorState = actor.y > 0 ? 'crossing_to_right' : 'crossing_to_left';
            }
          }
        }

        const moveSpeed = Math.hypot(actor.vx, actor.vy);
        if (moveSpeed > 0.04) {
          const targetHeading = Math.atan2(actor.vy, actor.vx);
          const angleDiff = Math.atan2(Math.sin(targetHeading - actor.heading), Math.cos(targetHeading - actor.heading));
          actor.heading += angleDiff * Math.min(1.0, dt * 6.0);

          // Articulated 4-leg walk cycle
          actor.walkPhase += dt * moveSpeed * 8.5;
          if (actor.mesh.userData.legPivots) {
            const pivots = actor.mesh.userData.legPivots;
            pivots[0].rotation.z = Math.sin(actor.walkPhase) * 0.38;
            pivots[1].rotation.z = -Math.sin(actor.walkPhase) * 0.38;
            pivots[2].rotation.z = -Math.sin(actor.walkPhase) * 0.38;
            pivots[3].rotation.z = Math.sin(actor.walkPhase) * 0.38;
          }
          actor.mesh.position.y = Math.abs(Math.sin(actor.walkPhase * 2.0)) * 0.035;

          if (actor.mesh.userData.tailPivot) {
            actor.mesh.userData.tailPivot.rotation.z = Math.sin(actor.walkPhase * 1.5) * 0.22;
          }
        } else {
          // Standing stance
          if (actor.mesh.userData.legPivots) {
            actor.mesh.userData.legPivots.forEach(p => { p.rotation.z *= 0.85; });
          }
          actor.mesh.position.y = 0;
          if (actor.mesh.userData.tailPivot) {
            actor.mesh.userData.tailPivot.rotation.z = Math.sin(state.simTime * 2.0) * 0.15;
          }
        }

        // Neck alert look angle towards approaching vehicle
        if (actor.mesh.userData.neckPivot) {
          if (distToEgo < 24.0) {
            actor.mesh.userData.neckPivot.rotation.y = Math.sin(state.simTime * 3.0) * 0.15 + (actor.y > ego.y ? -0.35 : 0.35);
          } else {
            actor.mesh.userData.neckPivot.rotation.y = 0;
          }
        }

        actor.x += actor.vx * dt;
        actor.y += actor.vy * dt;
        actor.mesh.position.set(actor.x, actor.mesh.position.y, actor.y);
        actor.mesh.rotation.y = -actor.heading;

      } else if (actor.type === 'motorcycle') {
        // Dynamic Indian Two-Wheeler: Unscripted Corridor Filtering, Overtaking & Banking
        actor.laneTimer = (actor.laneTimer || (2.0 + Math.random() * 3.0)) - dt;

        if (actor.laneTimer <= 0) {
          actor.laneTimer = 3.5 + Math.random() * 3.5;
          const corridors = actor.speed >= 0 ? [-2.2, -1.1, 0.0, 1.1, 2.2] : [1.4, 2.2, 3.0];
          actor.targetCorridor = corridors[Math.floor(Math.random() * corridors.length)];
        }

        // Scan ahead along bike's current path for obstacles to filter around
        for (let j = 0; j < state.trafficActors.length; j++) {
          if (i === j) continue;
          const other = state.trafficActors[j];
          const dX = (other.x - actor.x) * Math.sign(actor.speed);
          const dY = Math.abs(other.y - actor.y);

          if (dX > 0 && dX < 16.0 && dY < 1.35) {
            // Blocked by vehicle ahead: dynamically filter into adjacent open corridor
            actor.targetCorridor = actor.y > 0 ? (actor.targetCorridor - 1.2) : (actor.targetCorridor + 1.2);
            break;
          }
        }

        // Avoid Ego vehicle corridor
        const egoDx = (ego.x - actor.x) * Math.sign(actor.speed);
        if (egoDx > 0 && egoDx < 18.0 && Math.abs(ego.y - actor.y) < 1.35) {
          actor.targetCorridor = ego.y > 0 ? (ego.y - 1.4) : (ego.y + 1.4);
        }

        // Bound target corridor within drivable road width
        const desiredY = Math.max(-3.2, Math.min(3.2, actor.targetCorridor !== undefined ? actor.targetCorridor : actor.nominalY));
        const latDiff = desiredY - actor.y;
        actor.vy = Math.sign(latDiff) * Math.min(Math.abs(latDiff) * 3.0, 2.4);
        actor.y += actor.vy * dt;
        actor.x += actor.speed * dt;

        actor.mesh.position.set(actor.x, 0, actor.y);

        // Effective heading with dynamic lateral steering
        const baseH = actor.speed >= 0 ? 0 : Math.PI;
        const effHeading = baseH - Math.atan2(actor.vy, Math.max(2.0, Math.abs(actor.speed)));
        actor.mesh.rotation.y = -effHeading;

        // Realistic banking/lean angle when cornering
        if (actor.mesh.userData.rollGroup) {
          const bank = -Math.max(-0.45, Math.min(0.45, (actor.vy / Math.max(2.5, Math.abs(actor.speed))) * 1.5));
          actor.mesh.userData.rollGroup.rotation.x = bank;
        }

      } else {
        // Commercial Trucks & Auto-Rickshaws: Forward Sensing, Fixed Object Awareness & Standstill Buffer
        let leadObstacle = null;
        let leadDist = 999;

        // 1. Scan other traffic actors for mutual collision avoidance
        for (let j = 0; j < state.trafficActors.length; j++) {
          if (i === j) continue;
          const other = state.trafficActors[j];

          if (other.isCrossroadActor) {
            // Check if crossroad actor is crossing the roadway ahead
            if (Math.abs(other.y) < 4.8) {
              const aheadDist = (other.x - actor.x) * Math.sign(actor.speed);
              if (aheadDist > 0 && aheadDist < 26.0) {
                if (aheadDist < leadDist) {
                  leadDist = aheadDist;
                  leadObstacle = { speed: 0.0, type: 'crossroad', y: other.y };
                }
              }
            }
            continue;
          }

          // Check if 'other' is ahead along travel direction
          const aheadDist = (other.x - actor.x) * Math.sign(actor.speed);
          const latOverlap = Math.abs(actor.y - other.y) < (actor.width + other.width) * 0.65;

          if (aheadDist > 0 && aheadDist < 32.0 && latOverlap) {
            if (aheadDist < leadDist) {
              leadDist = aheadDist;
              leadObstacle = other;
            }
          }
        }

        // 2. Scan Ego Vehicle as a solid obstacle
        const egoAheadDist = (ego.x - actor.x) * Math.sign(actor.speed);
        if (egoAheadDist > 0 && egoAheadDist < 32.0 && Math.abs(actor.y - ego.y) < (actor.width + ego.width) * 0.65) {
          if (egoAheadDist < leadDist) {
            leadDist = egoAheadDist;
            leadObstacle = { speed: ego.v, type: 'ego', y: ego.y };
          }
        }

        // 3. Scan Surface Potholes as Road Surface Defects (Vehicles slow down and ROLL ACROSS, NOT a solid wall!)
        let approachingPothole = null;
        let potholeDist = 999;
        for (const pot of state.potholes) {
          const potAhead = (pot.x - actor.x) * Math.sign(actor.speed);
          if (potAhead > -2.0 && potAhead < 24.0 && Math.abs(actor.y - pot.y) < (actor.width * 0.5 + pot.radius + 0.25)) {
            if (potAhead < potholeDist) {
              potholeDist = potAhead;
              approachingPothole = pot;
            }
          }
        }

        // 4. Vehicle Speed Adaptation & Zero-Pushing Standstill Logic
        if (leadObstacle) {
          const isFixedObstacle = leadObstacle.speed < 1.0 || leadObstacle.type === 'cattle' || leadObstacle.type === 'crossroad';

          if (isFixedObstacle) {
            // Treat as SOLID FIXED OBJECT (cattle, crossroad vehicle, broken down car): stop before touching!
            if (leadDist < 4.5) {
              actor.targetSpeed = 0.0; // Standstill
            } else if (leadDist < 12.0) {
              actor.targetSpeed = Math.min(actor.targetSpeed, 2.0); // Hard braking
            } else {
              actor.targetSpeed = Math.min(actor.targetSpeed, 4.5); // Yielding approach
            }

            // Attempt lateral lane change if adjacent lane is open
            if (Math.abs(actor.y - leadObstacle.y) < 1.2 && leadObstacle.type !== 'crossroad') {
              actor.desiredY = actor.nominalY + (actor.y < 0 ? 1.6 : -1.6);
            }
          } else {
            // Moving vehicle ahead: strict standstill buffer prevents pushing!
            if (leadDist < 4.5) {
              actor.targetSpeed = 0.0; // Complete standstill - zero pushing
            } else if (leadDist < 10.0) {
              actor.targetSpeed = Math.max(0.0, Math.min(leadObstacle.speed * 0.7, (leadDist - 4.5) * 1.2));
            } else if (leadDist < 18.0) {
              actor.targetSpeed = Math.max(0.0, Math.min(actor.nominalSpeed, leadObstacle.speed * 0.95));
            } else {
              actor.targetSpeed = actor.nominalSpeed;
            }
          }
        } else {
          // Clear lane ahead
          actor.targetSpeed = actor.nominalSpeed;
          actor.desiredY = actor.nominalY;
        }

        // Dedicated Pothole Crossing: Slow down to caution crawl and roll right over the pothole!
        if (approachingPothole) {
          // Gentle lateral evasion nudge if room allows
          const nudge = actor.y >= approachingPothole.y ? 0.8 : -0.8;
          actor.desiredY = Math.max(-2.7, Math.min(2.7, actor.nominalY + nudge));

          // Crawl and roll across pothole without ever stopping like a wall!
          if (potholeDist > 10.0) {
            actor.targetSpeed = Math.min(actor.targetSpeed, 5.5); // Early deceleration
          } else if (potholeDist >= -1.0) {
            actor.targetSpeed = Math.min(actor.targetSpeed, 3.2); // Safe crawl pace: rolls across!
          }
          // Suspension jolt animation when rolling over the pothole
          if (potholeDist < 2.0 && potholeDist > -2.0) {
            actor.mesh.position.y = Math.sin(state.simTime * 28.0 + (actor.id || 1)) * 0.055;
          } else {
            actor.mesh.position.y = 0.0;
          }
        } else {
          actor.mesh.position.y = 0.0;
        }

        // Smooth speed adjustment
        actor.speed += Math.sign(actor.targetSpeed - actor.speed) * Math.min(Math.abs(actor.targetSpeed - actor.speed), 4.5 * dt);

        // Smooth lateral steering towards desired offset
        const latDiff = actor.desiredY - actor.y;
        actor.y += Math.sign(latDiff) * Math.min(Math.abs(latDiff), 1.4 * dt);

        actor.x += actor.speed * dt;
        actor.mesh.position.set(actor.x, 0, actor.y);
        actor.mesh.rotation.y = actor.speed >= 0 ? 0 : Math.PI;
      }

      // Recycle actors falling far behind
      if (actor.speed >= 0 && actor.x < ego.x - 48.0) {
        scene.remove(actor.mesh);
        state.trafficActors.splice(i, 1);
        continue;
      }
      if (actor.speed < 0 && actor.x < ego.x - 18.0) {
        scene.remove(actor.mesh);
        state.trafficActors.splice(i, 1);
        continue;
      }
    }

    // 2. Physical Anti-Clipping & Longitudinal Anti-Pushing Separation Pass (Includes Crossroad Traffic)
    for (let i = 0; i < state.trafficActors.length; i++) {
      const a = state.trafficActors[i];

      for (let j = i + 1; j < state.trafficActors.length; j++) {
        const b = state.trafficActors[j];

        // Scenario A: Both are crossroad vehicles
        if (a.isCrossroadActor && b.isCrossroadActor) {
          // If in same lane (both heading in same lateral direction):
          if (Math.sign(a.nominalVy) === Math.sign(b.nominalVy) && Math.abs(a.x - b.x) < 2.2) {
            const minGapY = (a.length + b.length) * 0.5 + 0.8;
            const distY = (b.y - a.y) * Math.sign(a.nominalVy);
            if (distY > 0 && distY < minGapY) {
              a.y = b.y - minGapY * Math.sign(a.nominalVy);
              a.vy = 0.0;
              a.mesh.position.set(a.x, 0, a.y);
            }
          }
          continue;
        }

        // Scenario B: One is crossroad vehicle and one is highway vehicle
        if (a.isCrossroadActor || b.isCrossroadActor) {
          const crossActor = a.isCrossroadActor ? a : b;
          const hwyActor = a.isCrossroadActor ? b : a;

          const dx = Math.abs(crossActor.x - hwyActor.x);
          const dy = Math.abs(crossActor.y - hwyActor.y);
          const minDx = (crossActor.width + hwyActor.length) * 0.48;
          const minDy = (crossActor.length + hwyActor.width) * 0.48;

          if (dx < minDx && dy < minDy) {
            // Collision resolution at intersection:
            if (Math.abs(crossActor.y) >= 4.6) {
              crossActor.y = Math.sign(crossActor.originY) * (4.8 + crossActor.length * 0.5 + 0.1);
              crossActor.vy = 0.0;
              crossActor.mesh.position.set(crossActor.x, 0, crossActor.y);
            } else if (hwyActor.speed > 0 && hwyActor.x < crossActor.x) {
              hwyActor.x = crossActor.x - minDx;
              hwyActor.speed = 0.0;
              hwyActor.mesh.position.set(hwyActor.x, 0, hwyActor.y);
            }
          }
          continue;
        }

        // Scenario C: Both are highway vehicles
        const minDx = (a.length + b.length) * 0.46;
        const minDy = (a.width + b.width) * 0.5;
        const curDx = Math.abs(a.x - b.x);
        const curDy = Math.abs(a.y - b.y);

        if (curDx < minDx && curDy < minDy) {
          const overlap = minDy - curDy + 0.05;
          if (a.y >= b.y) {
            a.y += overlap * 0.5;
            b.y -= overlap * 0.5;
          } else {
            a.y -= overlap * 0.5;
            b.y += overlap * 0.5;
          }
          a.mesh.position.z = a.y;
          b.mesh.position.z = b.y;
        }

        // Longitudinal Clamp Between Highway Traffic Actors (Zero Pushing)
        if (a.speed > 0 && b.x > a.x && Math.abs(a.y - b.y) < (a.width + b.width) * 0.5) {
          const minGap = (a.length + b.length) * 0.5 + 0.85;
          if (b.x - a.x < minGap) {
            a.x = b.x - minGap;
            a.speed = Math.min(a.speed, Math.max(0.0, b.speed));
            if (b.speed <= 0.1) a.speed = 0.0;
            a.mesh.position.x = a.x;
          }
        }
      }

      // Clamp against Ego Vehicle (Highway or Crossroad)
      if (a.isCrossroadActor) {
        const dx = Math.abs(a.x - ego.x);
        const dy = Math.abs(a.y - ego.y);
        const minDx = (a.width + ego.length) * 0.48;
        const minDy = (a.length + ego.width) * 0.48;

        if (dx < minDx && dy < minDy) {
          if (Math.abs(a.y) >= 4.6) {
            a.y = Math.sign(a.originY) * (4.8 + a.length * 0.5 + 0.1);
            a.vy = 0.0;
            a.mesh.position.set(a.x, 0, a.y);
          } else if (ego.v > 0 && ego.x < a.x) {
            ego.x = a.x - minDx;
            ego.v = 0.0;
          }
        }
      } else {
        const egoMinDx = (a.length + ego.length) * 0.46;
        const egoMinDy = (a.width + ego.width) * 0.5;
        const curEgoDx = Math.abs(a.x - ego.x);
        const curEgoDy = Math.abs(a.y - ego.y);

        if (curEgoDx < egoMinDx && curEgoDy < egoMinDy) {
          const overlap = egoMinDy - curEgoDy + 0.04;
          if (a.y >= ego.y) a.y += overlap;
          else a.y -= overlap;
          a.mesh.position.z = a.y;
        }

        // If actor is behind ego in same lane, clamp it behind ego (cannot push ego forward!)
        if (a.speed > 0 && ego.x > a.x && Math.abs(a.y - ego.y) < (a.width + ego.width) * 0.5) {
          const minGap = (a.length + ego.length) * 0.5 + 0.85;
          if (ego.x - a.x < minGap) {
            a.x = ego.x - minGap;
            a.speed = Math.min(a.speed, Math.max(0.0, ego.v));
            if (ego.v <= 0.1) a.speed = 0.0;
            a.mesh.position.x = a.x;
          }
        }
      }
    }

    // 3. Dynamic Traffic Spawner (Diverse, Unscripted Indian Traffic Flow)
    const targetCount = state.trafficDensity === 'heavy' ? 8 : (state.trafficDensity === 'medium' ? 5 : 2);
    if (state.trafficActors.length < targetCount) {
      const types = ['auto_rickshaw', 'motorcycle', 'cattle', 'truck'];
      const pick = types[Math.floor(Math.random() * types.length)];
      const aheadDist = 72 + Math.random() * 55;

      if (pick === 'cattle') {
        // Spawn from either shoulder or road margin
        const fromLeft = Math.random() > 0.5;
        const lat = fromLeft ? (3.6 + Math.random() * 0.8) : (-3.6 - Math.random() * 0.8);
        spawnActor('cattle', aheadDist, lat, 0.55, {
          mode: fromLeft ? 'crossing_to_right' : 'crossing_to_left'
        });
      } else if (pick === 'motorcycle') {
        const isOncoming = Math.random() > 0.35;
        const lat = isOncoming ? (0.8 + Math.random() * 1.8) : (Math.random() > 0.5 ? -1.4 : 1.4);
        const spd = isOncoming ? -(8.0 + Math.random() * 3.0) : (12.0 + Math.random() * 2.5);
        const dist = isOncoming ? aheadDist : (-30 - Math.random() * 15);
        spawnActor('motorcycle', dist, lat, spd);
      } else if (pick === 'truck') {
        const lat = Math.random() > 0.5 ? -1.8 : 1.8;
        spawnActor('truck', aheadDist, lat, 6.8 + Math.random() * 0.8);
      } else {
        const lat = Math.random() > 0.5 ? 1.6 : -1.6;
        spawnActor('auto_rickshaw', aheadDist, lat, 5.8 + Math.random() * 0.8);
      }
    }
  }

  // --- UPDATE TRAJECTORY RIBBONS ---
  function updateTrajectoryVisuals() {
    const ego = state.ego;

    // Active path
    const pts = [];
    for (let s = 0; s < 35; s += 1.5) {
      const tRatio = s / 35.0;
      const x = ego.x + s * Math.cos(ego.psi);
      const z = ego.y + (ego.targetD - ego.y) * Math.sin(tRatio * Math.PI * 0.5);
      pts.push(new THREE.Vector3(x, 0.12, z));
    }
    state.activePathMesh.geometry.setFromPoints(pts);

    // Candidate fan
    const candLines = [];
    [-2.6, -1.8, -0.9, 0.0, 0.9, 1.8, 2.6].forEach(dOffset => {
      for (let s = 0; s < 28; s += 2.5) {
        const x1 = ego.x + s;
        const z1 = ego.y + (dOffset - ego.y) * (s / 28.0);
        const x2 = ego.x + s + 2.5;
        const z2 = ego.y + (dOffset - ego.y) * ((s + 2.5) / 28.0);
        candLines.push(new THREE.Vector3(x1, 0.08, z1));
        candLines.push(new THREE.Vector3(x2, 0.08, z2));
      }
    });
    state.candidatePathsMesh.geometry.setFromPoints(candLines);
  }

  // --- 3D CAMERA SYSTEM ---
  function updateCamera() {
    const ego = state.ego;

    if (state.cameraView === 'chase') {
      controls.enabled = false;
      const joltOffset = ego.suspensionJolt > 0.002 ? Math.sin(state.simTime * 32.0) * ego.suspensionJolt * 0.45 : 0;
      const targetPos = new THREE.Vector3(
        ego.x - 13.5 * Math.cos(ego.psi),
        5.2 + joltOffset,
        ego.y - 13.5 * Math.sin(ego.psi)
      );
      camera.position.lerp(targetPos, 0.14);
      camera.lookAt(ego.x + 12.0 * Math.cos(ego.psi), 1.2, ego.y);
    }
    else if (state.cameraView === 'cockpit') {
      controls.enabled = false;
      const joltCockpit = ego.suspensionJolt > 0.002 ? Math.sin(state.simTime * 32.0) * ego.suspensionJolt * 0.35 : 0;
      camera.position.set(ego.x + 0.15, 1.35 + joltCockpit, ego.y);
      camera.lookAt(ego.x + 30.0 * Math.cos(ego.psi), 1.1, ego.y + 30.0 * Math.sin(ego.psi));
    }
    else if (state.cameraView === 'drone') {
      controls.enabled = false;
      camera.position.set(ego.x + 6.0, 36.0, ego.y);
      camera.lookAt(ego.x + 14.0, 0, ego.y);
    }
    else if (state.cameraView === 'orbit') {
      controls.enabled = true;
      controls.target.set(ego.x, 0.6, ego.y);
      controls.update();
    }
  }

  // --- HUD METERS & LIVE TELEMETRY ---
  function initHUD() {
    radarCanvas = document.getElementById('radarCanvas');
    radarCtx = radarCanvas.getContext('2d');
    sparklineCanvas = document.getElementById('sparklineCanvas');
    sparklineCtx = sparklineCanvas.getContext('2d');
  }

  function updateHUD() {
    const ego = state.ego;
    const speedKmh = ego.v * 3.6;
    const absSpeedKmh = Math.abs(speedKmh);

    // 1. Analog-Digital Speedometer
    if (speedVal) {
      if (ego.v < -0.15) {
        speedVal.textContent = `-${absSpeedKmh.toFixed(1)}`;
      } else {
        speedVal.textContent = absSpeedKmh.toFixed(1);
      }
    }
    if (targetSpeedVal) targetSpeedVal.textContent = `${(ego.targetV * 3.6).toFixed(1)} km/h`;

    // Dynamic Gear Indicator (P, R, N, D)
    const gearElP = document.getElementById('gearP');
    const gearElR = document.getElementById('gearR');
    const gearElN = document.getElementById('gearN');
    const gearElD = document.getElementById('gearD');
    if (gearElR && gearElD) {
      const activeGear = (ego.v < -0.15 || ego.gear === 'R') ? 'R' : (Math.abs(ego.v) <= 0.15 ? (state.mode === 'manual' ? 'P' : 'D') : 'D');
      [gearElP, gearElR, gearElN, gearElD].forEach(el => {
        if (!el) return;
        el.className = 'px-1.5 py-0.5 rounded font-bold text-slate-500 bg-slate-900 border border-slate-800';
      });
      if (activeGear === 'R' && gearElR) {
        gearElR.className = 'px-1.5 py-0.5 rounded font-bold text-amber-300 bg-amber-950 border border-amber-500/80 shadow-[0_0_10px_rgba(245,158,11,0.6)] animate-pulse';
      } else if (activeGear === 'D' && gearElD) {
        gearElD.className = 'px-1.5 py-0.5 rounded font-bold text-emerald-400 bg-emerald-950 border border-emerald-500/80 shadow-[0_0_10px_rgba(16,185,129,0.4)]';
      } else if (gearElP) {
        gearElP.className = 'px-1.5 py-0.5 rounded font-bold text-cyan-400 bg-cyan-950 border border-cyan-500/80';
      }
    }

    const needleDeg = -90 + (absSpeedKmh / 100) * 180;
    const needleRad = (needleDeg * Math.PI) / 180;
    const nx = 100 + 70 * Math.cos(needleRad);
    const ny = 105 + 70 * Math.sin(needleRad);
    if (speedNeedle) {
      speedNeedle.setAttribute('x2', nx.toFixed(1));
      speedNeedle.setAttribute('y2', ny.toFixed(1));
    }

    const arcCircum = 235.6;
    const arcOffset = arcCircum - (absSpeedKmh / 100) * arcCircum;
    if (speedArc) {
      speedArc.setAttribute('stroke-dashoffset', Math.max(0, arcOffset).toFixed(1));
    }

    // 2. Longitudinal G-Meter Bar
    if (accelVal) accelVal.textContent = `${ego.a >= 0 ? '+' : ''}${ego.a.toFixed(2)} m/s²`;
    if (accelBar) {
      const gNorm = ego.a >= 0 ? (ego.a / 3.2) * 50 : (ego.a / 6.5) * 50;
      if (ego.a >= 0) {
        accelBar.style.marginLeft = '50%';
        accelBar.style.width = `${Math.min(50, gNorm)}%`;
        accelBar.style.background = '#06b6d4';
      } else {
        accelBar.style.marginLeft = `${50 + gNorm}%`;
        accelBar.style.width = `${Math.abs(gNorm)}%`;
        accelBar.style.background = '#ef4444';
      }
    }

    // 3. Rotating Steering Wheel & Pose
    const steerDeg = (ego.delta * 180) / Math.PI;
    if (steeringWheelSvg) steeringWheelSvg.style.transform = `rotate(${-steerDeg * 3.5}deg)`;
    if (steerDegBadge) steerDegBadge.textContent = `${steerDeg >= 0 ? '+' : ''}${steerDeg.toFixed(1)}°`;
    if (headingVal) headingVal.textContent = `${((ego.psi * 180) / Math.PI).toFixed(1)}°`;
    if (distanceTraveledVal) distanceTraveledVal.textContent = `${state.distanceTraveled.toFixed(1)} m`;
    if (latDevVal) latDevVal.textContent = `${ego.y >= 0 ? '+' : ''}${ego.y.toFixed(2)} m`;

    // 4. Radar & Risk Matrix
    if (ttcVal) {
      ttcVal.textContent = ego.minTTC < 90 ? `${ego.minTTC.toFixed(1)}s` : 'CLEAR';
      ttcVal.className = `text-xl font-display font-bold ${ego.minTTC < 2.5 ? 'text-red-400' : 'text-emerald-400'}`;
    }
    if (clearanceVal) clearanceVal.textContent = ego.minClearance < 90 ? `${ego.minClearance.toFixed(1)} m` : '-- m';
    if (closestHazardVal) closestHazardVal.textContent = ego.closestHazard;

    if (riskLevelBadge) {
      riskLevelBadge.textContent = `${ego.riskLevel} RISK`;
      riskLevelBadge.className = `status-pill status-pill-${ego.riskLevel.toLowerCase()}`;
    }

    // 5. Frenet Decision States
    if (maneuverVal) maneuverVal.textContent = ego.activeManeuver.toUpperCase();
    if (plannerStateVal) plannerStateVal.textContent = ego.plannerState;
    if (replanCountVal) replanCountVal.textContent = `REPLANS: ${state.stats.replans}`;
    if (latencyVal) latencyVal.textContent = `${(44.0 + Math.random() * 8.0).toFixed(1)} ms`;

    // 6. Run Statistics
    if (collisionCountVal) collisionCountVal.textContent = state.stats.collisions;
    if (nearMissCountVal) nearMissCountVal.textContent = state.stats.nearMisses;
    if (simTimeVal) simTimeVal.textContent = `${state.simTime.toFixed(1)}s`;
    const avgSpd = state.simTime > 0 ? (state.distanceTraveled / state.simTime) * 3.6 : 0;
    if (avgSpeedVal) avgSpeedVal.textContent = `${avgSpd.toFixed(1)} km/h`;

    // 7. Radar Canvas Sweep
    drawTacticalRadar();

    // 8. Sparkline Telemetry Graph
    drawSparkline(speedKmh, ego.minClearance);
  }

  function drawTacticalRadar() {
    const ego = state.ego;
    const w = radarCanvas.width;
    const h = radarCanvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const maxRange = 50.0;

    radarCtx.clearRect(0, 0, w, h);

    // Radar Circles
    radarCtx.strokeStyle = 'rgba(6, 182, 212, 0.25)';
    radarCtx.lineWidth = 1;
    [0.33, 0.66, 1.0].forEach(rRatio => {
      radarCtx.beginPath();
      radarCtx.arc(cx, cy, (w / 2 - 4) * rRatio, 0, Math.PI * 2);
      radarCtx.stroke();
    });

    // Crosshairs
    radarCtx.beginPath();
    radarCtx.moveTo(cx, 4);
    radarCtx.lineTo(cx, h - 4);
    radarCtx.moveTo(4, cy);
    radarCtx.lineTo(w - 4, cy);
    radarCtx.stroke();

    // Upcoming Crossroad Junction Line
    const nextCrossX = getUpcomingCrossroadX(0);
    const crossDx = nextCrossX - ego.x;
    if (crossDx > -10 && crossDx < maxRange) {
      const crossBy = cy - (crossDx / maxRange) * (h / 2 - 6);
      radarCtx.save();
      radarCtx.setLineDash([3, 2]);
      radarCtx.strokeStyle = 'rgba(245, 158, 11, 0.85)'; // Amber dashed crossroad indicator
      radarCtx.lineWidth = 2;
      radarCtx.beginPath();
      radarCtx.moveTo(cx - 30, crossBy);
      radarCtx.lineTo(cx + 30, crossBy);
      radarCtx.stroke();
      radarCtx.restore();
    }

    // Pothole Hazard Rings (hollow rings matching defect radius)
    for (const pot of state.potholes) {
      const dx = pot.x - ego.x;
      const dy = pot.y - ego.y;
      if (Math.abs(dx) > maxRange || Math.abs(dy) > maxRange) continue;

      const bx = cx - (dy / maxRange) * (w / 2 - 6);
      const by = cy - (dx / maxRange) * (h / 2 - 6);

      radarCtx.strokeStyle = pot.hit ? 'rgba(239, 68, 68, 0.9)' : 'rgba(244, 63, 94, 0.85)';
      radarCtx.lineWidth = 1.5;
      radarCtx.beginPath();
      radarCtx.arc(bx, by, Math.max(2.5, pot.radius * 3.0), 0, Math.PI * 2);
      radarCtx.stroke();
    }

    // Obstacle Blips
    for (const actor of state.trafficActors) {
      const dx = actor.x - ego.x;
      const dy = actor.y - ego.y;
      if (Math.abs(dx) > maxRange || Math.abs(dy) > maxRange) continue;

      const bx = cx - (dy / maxRange) * (w / 2 - 6);
      const by = cy - (dx / maxRange) * (h / 2 - 6);

      radarCtx.fillStyle = actor.isCrossroadActor ? '#f97316' : (actor.type === 'cattle' ? '#f59e0b' : (actor.type === 'truck' ? '#ef4444' : '#00f2fe'));
      radarCtx.shadowColor = radarCtx.fillStyle;
      radarCtx.shadowBlur = 6;
      radarCtx.beginPath();
      radarCtx.arc(bx, by, 3.5, 0, Math.PI * 2);
      radarCtx.fill();
      radarCtx.shadowBlur = 0;
    }

    // Rotating Radar Sweep Hand
    const sweepAngle = (state.simTime * 4.0) % (Math.PI * 2);
    radarCtx.strokeStyle = 'rgba(0, 242, 254, 0.6)';
    radarCtx.lineWidth = 1.2;
    radarCtx.beginPath();
    radarCtx.moveTo(cx, cy);
    radarCtx.lineTo(cx + Math.cos(sweepAngle) * (w / 2 - 4), cy + Math.sin(sweepAngle) * (h / 2 - 4));
    radarCtx.stroke();

    // Center Ego Vehicle Blip & Heading Pointer
    radarCtx.fillStyle = '#10b981';
    radarCtx.beginPath();
    radarCtx.arc(cx, cy, 3.5, 0, Math.PI * 2);
    radarCtx.fill();

    radarCtx.strokeStyle = '#10b981';
    radarCtx.lineWidth = 2;
    radarCtx.beginPath();
    radarCtx.moveTo(cx, cy);
    radarCtx.lineTo(cx, cy - 7);
    radarCtx.stroke();
  }

  function drawSparkline(speed, clearance) {
    state.stats.speedHistory.push(speed);
    state.stats.clearanceHistory.push(clearance);
    if (state.stats.speedHistory.length > 75) {
      state.stats.speedHistory.shift();
      state.stats.clearanceHistory.shift();
    }

    const w = sparklineCanvas.width;
    const h = sparklineCanvas.height;
    sparklineCtx.clearRect(0, 0, w, h);

    // Speed curve (cyan)
    sparklineCtx.beginPath();
    sparklineCtx.strokeStyle = '#00f2fe';
    sparklineCtx.lineWidth = 1.5;
    for (let i = 0; i < state.stats.speedHistory.length; i++) {
      const x = (i / 75) * w;
      const y = h - (state.stats.speedHistory[i] / 80) * (h - 4) - 2;
      if (i === 0) sparklineCtx.moveTo(x, y);
      else sparklineCtx.lineTo(x, y);
    }
    sparklineCtx.stroke();

    // Clearance curve (green)
    sparklineCtx.beginPath();
    sparklineCtx.strokeStyle = '#10b981';
    sparklineCtx.lineWidth = 1.5;
    for (let i = 0; i < state.stats.clearanceHistory.length; i++) {
      const x = (i / 75) * w;
      const y = h - (Math.min(15, state.stats.clearanceHistory[i]) / 15) * (h - 4) - 2;
      if (i === 0) sparklineCtx.moveTo(x, y);
      else sparklineCtx.lineTo(x, y);
    }
    sparklineCtx.stroke();
  }

  function addEventLog(msg, colorClass = 'text-slate-300') {
    const div = document.createElement('div');
    div.className = colorClass;
    div.textContent = msg;
    eventTicker.appendChild(div);
    if (eventTicker.children.length > 5) {
      eventTicker.removeChild(eventTicker.children[0]);
    }
  }

  // --- WEATHER & LIGHTING MODES ---
  function updateWeather(type) {
    state.weather = type;
    if (type === 'sunset') {
      scene.background.set(0xf97316);
      hemiLight.color.set(0xfb923c);
      dirLight.color.set(0xf97316);
      dirLight.intensity = 1.6;
      fogObj.color.set(0xf97316);
      fogObj.density = 0.005;
      rainParticleSystem.visible = false;
    } else if (type === 'night') {
      scene.background.set(0x020617);
      hemiLight.color.set(0x1e293b);
      dirLight.color.set(0x38bdf8);
      dirLight.intensity = 0.25;
      fogObj.color.set(0x020617);
      fogObj.density = 0.007;
      rainParticleSystem.visible = false;
    } else if (type === 'rain') {
      scene.background.set(0x334155);
      hemiLight.color.set(0x64748b);
      dirLight.color.set(0x94a3b8);
      dirLight.intensity = 0.8;
      fogObj.color.set(0x334155);
      fogObj.density = 0.01;
      rainParticleSystem.visible = true;
    } else if (type === 'fog') {
      scene.background.set(0x94a3b8);
      hemiLight.color.set(0xcbd5e1);
      dirLight.intensity = 0.5;
      fogObj.color.set(0x94a3b8);
      fogObj.density = 0.022;
      rainParticleSystem.visible = false;
    } else {
      // Clear Daylight
      scene.background.set(0x87ceeb);
      hemiLight.color.set(0xffffff);
      hemiLight.intensity = 1.05;
      dirLight.color.set(0xfff8ee);
      dirLight.intensity = 1.45;
      fogObj.color.set(0xcfe4fa);
      fogObj.density = 0.0032;
      rainParticleSystem.visible = false;
    }
  }

  // --- USER CONTROLS & INTERACTION ---
  function initEventListeners() {
    const camButtons = {
      camChaseBtn: 'chase',
      camCockpitBtn: 'cockpit',
      camDroneBtn: 'drone',
      camOrbitBtn: 'orbit'
    };
    Object.entries(camButtons).forEach(([id, view]) => {
      document.getElementById(id).addEventListener('click', () => {
        Object.keys(camButtons).forEach(bId => document.getElementById(bId).classList.remove('active'));
        document.getElementById(id).classList.add('active');
        state.cameraView = view;
      });
    });

    const modeAutoBtn = document.getElementById('modeAutoBtn');
    const modeManualBtn = document.getElementById('modeManualBtn');
    const modeReverseBtn = document.getElementById('modeReverseBtn');

    function setActiveMode(mode) {
      state.mode = mode;
      modeAutoBtn.classList.toggle('active', mode === 'auto');
      modeManualBtn.classList.toggle('active', mode === 'manual');
      if (modeReverseBtn) modeReverseBtn.classList.toggle('active', mode === 'reverse');

      if (mode === 'auto') {
        state.ego.isReversing = false;
        state.ego.reverseTimer = 0.0;
        state.ego.deadlockTimer = 0.0;
        state.ego.gear = 'D';
        vehicleModeBadge.textContent = 'AUTONOMOUS';
        vehicleModeBadge.className = 'px-1.5 py-0.5 text-[10px] font-mono font-bold rounded bg-emerald-950/80 text-emerald-300 border border-emerald-500/30';
        addEventLog('Autonomous Lattice AI engaged.', 'text-emerald-400');
      } else if (mode === 'manual') {
        state.ego.isReversing = false;
        state.ego.reverseTimer = 0.0;
        state.ego.deadlockTimer = 0.0;
        vehicleModeBadge.textContent = 'MANUAL (WASD)';
        vehicleModeBadge.className = 'px-1.5 py-0.5 text-[10px] font-mono font-bold rounded bg-amber-950/80 text-amber-300 border border-amber-500/30';
        addEventLog('Manual WASD override active.', 'text-amber-300');
      } else if (mode === 'reverse') {
        state.ego.isReversing = true;
        state.ego.gear = 'R';
        state.ego.targetV = -2.2;
        state.ego.targetDelta = 0.0;
        vehicleModeBadge.textContent = 'REVERSE [R]';
        vehicleModeBadge.className = 'px-1.5 py-0.5 text-[10px] font-mono font-bold rounded bg-purple-950/80 text-purple-300 border border-purple-500/30';
        addEventLog('⏪ REVERSE MODE: Backing up vehicle safely. Rear collision radar active.', 'text-purple-300 font-bold');
      }
    }

    modeAutoBtn.addEventListener('click', () => setActiveMode('auto'));
    modeManualBtn.addEventListener('click', () => setActiveMode('manual'));
    if (modeReverseBtn) {
      modeReverseBtn.addEventListener('click', () => setActiveMode('reverse'));
    }

    document.getElementById('weatherSelect').addEventListener('change', e => {
      updateWeather(e.target.value);
    });

    document.getElementById('trafficDensitySelect').addEventListener('change', e => {
      state.trafficDensity = e.target.value;
    });

    // Dynamic Hazard Trigger Buttons (Unscripted & Multi-Corner)
    document.getElementById('spawnCattleBtn').addEventListener('click', () => {
      const fromLeft = Math.random() > 0.5;
      const lat = fromLeft ? (3.6 + Math.random() * 0.8) : (-3.6 - Math.random() * 0.8);
      const aheadX = 38 + Math.random() * 15;
      spawnActor('cattle', aheadX, lat, 0.6, {
        mode: fromLeft ? 'crossing_to_right' : 'crossing_to_left'
      });
      addEventLog(`⚠️ HAZARD TRIGGER: Stray Cow entering from ${fromLeft ? 'left shoulder' : 'right shoulder'}!`, 'text-amber-400 font-bold');
    });

    document.getElementById('spawnRickshawBtn').addEventListener('click', () => {
      const lat = (Math.random() > 0.5 ? 1.6 : -1.6) + (Math.random() - 0.5) * 0.6;
      spawnActor('auto_rickshaw', 32 + Math.random() * 12, lat, 5.8);
      addEventLog('⚠️ HAZARD TRIGGER: Auto-Rickshaw in dynamic traffic flow ahead!', 'text-yellow-400 font-bold');
    });

    document.getElementById('spawnBikeBtn').addEventListener('click', () => {
      const isOncoming = Math.random() > 0.35;
      const lat = isOncoming ? (0.8 + Math.random() * 1.8) : (Math.random() > 0.5 ? -1.4 : 1.4);
      const spd = isOncoming ? -(8.0 + Math.random() * 3.0) : (12.0 + Math.random() * 3.0);
      const dist = isOncoming ? (65 + Math.random() * 20) : (-25 - Math.random() * 15);
      spawnActor('motorcycle', dist, lat, spd);
      addEventLog(isOncoming ? '⚠️ HAZARD TRIGGER: Oncoming Motorcycle filtering dynamically!' : '⚠️ HAZARD TRIGGER: Fast Motorcycle overtaking from rear!', 'text-purple-400 font-bold');
    });

    document.getElementById('spawnPotholeBtn').addEventListener('click', () => {
      // Spawn directly ahead in ego car's current lane path for active braking reaction
      spawnPothole(28, state.ego.y, 0.85);
      addEventLog('⚠️ HAZARD TRIGGER: Deep surface pothole ahead! Active braking & evasion engaged.', 'text-red-400 font-bold');
    });

    document.getElementById('spawnTruckBtn').addEventListener('click', () => {
      const lat = Math.random() > 0.5 ? -1.8 : 1.8;
      spawnActor('truck', 42 + Math.random() * 14, lat, 6.8);
      addEventLog('⚠️ HAZARD TRIGGER: Heavy Commercial Freight Lorry ahead!', 'text-orange-400 font-bold');
    });

    const spawnCrossroadBtn = document.getElementById('spawnCrossroadBtn');
    if (spawnCrossroadBtn) {
      spawnCrossroadBtn.addEventListener('click', () => {
        const nextJunctionX = getUpcomingCrossroadX(18);
        spawnCrossroadActor(nextJunctionX);
        addEventLog(`⚠️ HAZARD TRIGGER: Crossroad traffic entering intersection at x=${nextJunctionX.toFixed(0)}m!`, 'text-emerald-400 font-bold');
      });
    }

    const pauseBtn = document.getElementById('pauseSimBtn');
    const pauseIcon = document.getElementById('pauseSimIcon');
    const playIcon = document.getElementById('playSimIcon');
    pauseBtn.addEventListener('click', () => {
      state.isPaused = !state.isPaused;
      if (state.isPaused) {
        pauseIcon.classList.add('hidden');
        playIcon.classList.remove('hidden');
      } else {
        pauseIcon.classList.remove('hidden');
        playIcon.classList.add('hidden');
      }
    });

    document.getElementById('resetSimBtn').addEventListener('click', () => {
      state.ego.x = 0;
      state.ego.y = 0;
      state.ego.v = 12.0;
      state.ego.psi = 0.0;
      state.ego.delta = 0.0;
      state.ego.a = 0.0;
      state.ego.targetD = 0.0;
      state.ego.suspensionJolt = 0.0;
      state.ego.pitchJolt = 0.0;
      state.ego.gear = 'D';
      state.ego.isReversing = false;
      state.ego.reverseTimer = 0.0;
      state.ego.deadlockTimer = 0.0;
      state.simTime = 0;
      state.distanceTraveled = 0;
      state.lastPotholeSpawnX = 0;
      state.lastCrossroadSpawnX = 0;
      state.roadSegments.forEach((seg, i) => {
        seg.position.x = i * 120;
      });
      populateInitialEnvironment();
      setActiveMode('auto');
      addEventLog('Simulation reset to initial state.', 'text-cyan-300');
    });

    window.addEventListener('keydown', e => {
      if (e.code === 'KeyW' || e.code === 'ArrowUp') state.keys.forward = true;
      if (e.code === 'KeyS' || e.code === 'ArrowDown') state.keys.backward = true;
      if (e.code === 'KeyA' || e.code === 'ArrowLeft') state.keys.left = true;
      if (e.code === 'KeyD' || e.code === 'ArrowRight') state.keys.right = true;
      if (e.code === 'Space') {
        e.preventDefault();
        pauseBtn.click();
      }
      if (e.code === 'Digit1') document.getElementById('camChaseBtn').click();
      if (e.code === 'Digit2') document.getElementById('camCockpitBtn').click();
      if (e.code === 'Digit3') document.getElementById('camDroneBtn').click();
      if (e.code === 'Digit4') document.getElementById('camOrbitBtn').click();
    });

    window.addEventListener('keyup', e => {
      if (e.code === 'KeyW' || e.code === 'ArrowUp') state.keys.forward = false;
      if (e.code === 'KeyS' || e.code === 'ArrowDown') state.keys.backward = false;
      if (e.code === 'KeyA' || e.code === 'ArrowLeft') state.keys.left = false;
      if (e.code === 'KeyD' || e.code === 'ArrowRight') state.keys.right = false;
    });
  }

  // --- ANIMATION / RENDER LOOP ---
  let lastTime = 0;
  function animate(time) {
    requestAnimationFrame(animate);

    const dt = Math.min(0.05, (time - lastTime) / 1000.0 || 0.016);
    lastTime = time;

    updateSimulationPhysics(dt);
    updateHUD();
    renderer.render(scene, camera);
  }

  window.addEventListener('DOMContentLoaded', init);
})();
