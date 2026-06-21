/* =====================================================================
 * effects.js — "juice" / game-feel layer for the dojo replay viewer
 * ---------------------------------------------------------------------
 * Self-contained, no build step, no deps. Plain canvas-2D primitives that
 * compose with any render loop. Everything is driven by a wall-clock time
 * `t` (milliseconds) so effects animate independently of the replay frame.
 *
 * Drop-in usage from a host render loop:
 *
 *   const fx = window.DojoFX.create();      // one instance per viewer
 *   ...
 *   function frame(now) {                    // now = performance.now()
 *     const dt = fx.tick(now);               // advances internal clock
 *
 *     ctx.save();
 *     fx.applyShake(ctx);                    // <- camera shake wraps the scene
 *     drawSceneAndFighters();                // host's existing draw
 *     fx.drawWorld(ctx, now);                // streaks + impacts + puffs (world space)
 *     ctx.restore();
 *
 *     fx.ambient(ctx, now, { groundY });     // vignette, lanterns, dust (screen space)
 *     fx.drawHud(ctx, now);                  // RING OUT! / K.O. stinger (screen space)
 *   }
 *
 * Triggering (call once at the moment the event happens):
 *   fx.spawnImpact(x, y, { facing, color });   // punch landed
 *   fx.knockback(x, y, { dir, color });        // fighter launched
 *   fx.spawnDust(x, groundY, { dir });         // landing / dash puff
 *   fx.ringOut(winner, { color });             // K.O. stinger + flash
 *
 * Pure helpers (no instance needed) are also exported for ad-hoc use:
 *   DojoFX.spawnImpact(ctx, x, y, opts)   — draws a single static impact star
 *   DojoFX.ringOutStinger(ctx, winner, p) — draws the stinger at progress p∈[0,1]
 *   DojoFX.ambient(ctx, t, opts)          — stateless ambient pass
 *   DojoFX.shake(intensity)               — global convenience (uses a default instance)
 * ===================================================================== */
(function (root) {
  "use strict";

  // ----- dojo palette (mirrors viewer.html tokens) --------------------
  const PAL = {
    ink:     [239, 230, 214],   // --ink
    paper:   [236, 225, 200],   // --paper
    orange:  [224, 134, 58],    // --orange  (brand)
    gold:    [200, 151, 46],    // --gold
    lacquer: [138, 42, 32],     // --lacquer (deep red)
    ember:   [255, 196, 120],   // warm lantern core
    smoke:   [120, 102, 80],    // --muted2-ish dust
    trained: [196, 69, 60],     // --trained (red side)
    base:    [74, 99, 176],     // --base    (blue side)
  };

  const TAU = Math.PI * 2;
  const rand = (a, b) => a + Math.random() * (b - a);
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const lerp = (a, b, p) => a + (b - a) * p;
  // ease-out cubic — most juice wants a fast attack then a long tail
  const easeOut = (p) => 1 - Math.pow(1 - p, 3);
  const easeIn = (p) => p * p;
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

  // ===================================================================
  //  Stateless primitives — take a ctx + explicit progress. These never
  //  read internal state, so they can be reused frame-by-frame or baked.
  // ===================================================================

  /* A single impact star + radial burst at (x,y). `p`∈[0,1] is the life
   * progress (0 = freshest, 1 = gone). `dir` biases the spray. */
  function drawImpact(ctx, x, y, p, opts) {
    opts = opts || {};
    const dir = opts.dir != null ? opts.dir : 0;        // -1 / +1 horizontal bias
    const scale = opts.scale != null ? opts.scale : 1;
    const tint = opts.color || PAL.orange;
    const e = easeOut(clamp(p, 0, 1));
    const fade = 1 - e;
    if (fade <= 0.001) return;

    ctx.save();
    ctx.translate(x, y);
    ctx.globalCompositeOperation = "lighter";

    // (1) bloom — soft warm flash that expands and dies fast
    const bloomR = (10 + e * 46) * scale;
    const bloom = ctx.createRadialGradient(0, 0, 0, 0, 0, bloomR);
    bloom.addColorStop(0, rgba(PAL.ember, 0.55 * fade));
    bloom.addColorStop(0.4, rgba(tint, 0.30 * fade));
    bloom.addColorStop(1, rgba(tint, 0));
    ctx.fillStyle = bloom;
    ctx.beginPath(); ctx.arc(0, 0, bloomR, 0, TAU); ctx.fill();

    // (2) shock ring — thin expanding circle, the "pop"
    const ringR = (6 + easeOut(clamp(p * 1.15, 0, 1)) * 40) * scale;
    ctx.lineWidth = clamp(3 * fade, 0.4, 3) * scale;
    ctx.strokeStyle = rgba(PAL.ink, 0.5 * fade);
    ctx.beginPath(); ctx.arc(0, 0, ringR, 0, TAU); ctx.stroke();

    // (3) impact star — sharp asymmetric spokes (manga "POW")
    const spokes = 7;
    const inner = 3 * scale;
    const reach = (16 + e * 14) * scale;
    ctx.fillStyle = rgba(PAL.ink, 0.9 * fade);
    ctx.beginPath();
    for (let i = 0; i < spokes; i++) {
      const a = (i / spokes) * TAU + (opts.seed || 0);
      // jagged: alternate long/short, length jittered deterministically by i
      const long = reach * (0.7 + 0.5 * ((i * 53) % 7) / 7);
      const r1 = long + dir * Math.cos(a) * 6 * scale;
      ctx.lineTo(Math.cos(a) * r1, Math.sin(a) * r1);
      const a2 = a + TAU / spokes / 2;
      ctx.lineTo(Math.cos(a2) * inner, Math.sin(a2) * inner);
    }
    ctx.closePath();
    ctx.fill();

    // (4) hot core dot
    ctx.fillStyle = rgba([255, 244, 224], 0.95 * fade);
    ctx.beginPath(); ctx.arc(0, 0, 3.2 * scale * fade, 0, TAU); ctx.fill();

    ctx.restore();
  }

  /* The RING OUT! / K.O. stinger. `p`∈[0,1] progress through its life.
   * `winner` is a label string (or null). Draws full-screen, screen space. */
  function ringOutStinger(ctx, winner, p, opts) {
    opts = opts || {};
    const W = opts.W != null ? opts.W : ctx.canvas.width;
    const H = opts.H != null ? opts.H : ctx.canvas.height;
    const tint = opts.color || PAL.orange;
    p = clamp(p, 0, 1);

    ctx.save();

    // (1) impact white-out flash on the leading edge, then a held dim wash
    const flash = p < 0.12 ? (1 - p / 0.12) : 0;
    if (flash > 0) {
      ctx.fillStyle = rgba([255, 250, 240], 0.85 * flash);
      ctx.fillRect(0, 0, W, H);
    }
    // darken the rest of the scene so the text reads
    const wash = clamp((p - 0.05) * 6, 0, 1) * (p > 0.85 ? (1 - p) / 0.15 : 1);
    ctx.fillStyle = rgba([8, 6, 4], 0.5 * wash);
    ctx.fillRect(0, 0, W, H);

    // (2) speed lines converging on center — manga finisher energy
    const cx = W / 2, cy = H * 0.46;
    const lineP = clamp(p * 2.2, 0, 1);
    if (wash > 0.02) {
      ctx.save();
      ctx.translate(cx, cy);
      ctx.globalAlpha = 0.5 * wash;
      ctx.strokeStyle = rgba(PAL.ink, 0.4);
      ctx.lineWidth = 2;
      const N = 40;
      const maxR = Math.hypot(W, H);
      for (let i = 0; i < N; i++) {
        const a = (i / N) * TAU + (i % 2) * 0.04;
        const r0 = lerp(maxR, maxR * 0.32, lineP) * (0.8 + 0.2 * ((i * 17) % 5) / 5);
        const r1 = maxR;
        ctx.lineWidth = 1.5 + ((i * 7) % 3);
        ctx.beginPath();
        ctx.moveTo(Math.cos(a) * r0, Math.sin(a) * r0);
        ctx.lineTo(Math.cos(a) * r1, Math.sin(a) * r1);
        ctx.stroke();
      }
      ctx.restore();
    }

    // (3) the stinger word — "K.O." punches in, then "RING OUT!" subtitle
    const popP = easeOut(clamp(p * 3.2, 0, 1));
    const settle = 1 + (1 - popP) * 0.6;          // overshoot scale-down
    const drift = (1 - clamp((p - 0.6) / 0.4, 0, 1));
    const textA = clamp(p > 0.82 ? (1 - p) / 0.18 : 1, 0, 1);

    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(settle, settle);
    ctx.globalAlpha = textA;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    // big KO — paper-cream with a lacquer slab shadow + gold edge
    const big = Math.round(H * 0.20);
    ctx.font = `900 ${big}px ui-sans-serif, "Hiragino Sans", system-ui, sans-serif`;
    ctx.lineJoin = "round";
    // back slab
    ctx.fillStyle = rgba(PAL.lacquer, 0.9);
    ctx.fillText("K.O.", 5, 6);
    // body
    ctx.lineWidth = big * 0.06;
    ctx.strokeStyle = rgba([20, 14, 9], 0.9);
    ctx.strokeText("K.O.", 0, 0);
    ctx.fillStyle = rgba(PAL.paper, 1);
    ctx.fillText("K.O.", 0, 0);
    // gold gleam sweeping across as it settles
    const gleam = clamp((p - 0.18) * 3, 0, 1);
    if (gleam > 0 && gleam < 1) {
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      const gx = lerp(-big * 1.6, big * 1.6, gleam);
      const g = ctx.createLinearGradient(gx - big * 0.4, 0, gx + big * 0.4, 0);
      g.addColorStop(0, rgba(PAL.gold, 0));
      g.addColorStop(0.5, rgba(PAL.gold, 0.8));
      g.addColorStop(1, rgba(PAL.gold, 0));
      ctx.fillStyle = g;
      ctx.fillText("K.O.", 0, 0);
      ctx.restore();
    }

    // subtitle: RING OUT! — letter-spaced, orange, slides up
    const sub = Math.round(H * 0.052);
    ctx.font = `800 ${sub}px ui-sans-serif, system-ui, sans-serif`;
    ctx.globalAlpha = textA * clamp(p * 4, 0, 1);
    const subY = big * 0.62 + drift * 14;
    spacedText(ctx, "R I N G   O U T", 0, subY, rgba(tint, 1), rgba([20, 14, 9], 0.85), sub * 0.10);

    // winner tag, smaller, beneath
    if (winner) {
      const wf = Math.round(H * 0.030);
      ctx.font = `600 ${wf}px ui-sans-serif, system-ui, sans-serif`;
      ctx.globalAlpha = textA * clamp((p - 0.15) * 4, 0, 1);
      ctx.fillStyle = rgba(PAL.ink, 0.85);
      ctx.fillText(String(winner).toUpperCase() + "  WINS", 0, subY + sub * 1.4);
    }
    ctx.restore();

    ctx.restore();
  }

  // tiny helper: stroked+filled centered text with manual tracking
  function spacedText(ctx, str, x, y, fill, stroke, track) {
    const chars = str.split("");
    const widths = chars.map((c) => ctx.measureText(c).width + (c === " " ? 0 : track));
    let total = widths.reduce((a, b) => a + b, 0) - track;
    let cx = x - total / 2;
    ctx.save();
    ctx.textAlign = "left";
    ctx.lineWidth = Math.max(2, ctx.measureText("M").width * 0.04);
    for (let i = 0; i < chars.length; i++) {
      ctx.strokeStyle = stroke; ctx.strokeText(chars[i], cx, y);
      ctx.fillStyle = fill; ctx.fillText(chars[i], cx, y);
      cx += widths[i];
    }
    ctx.restore();
  }

  /* Stateless ambient atmosphere pass: vignette + warm glow + drifting
   * dust motes + swaying hanging lanterns + a faint floor light pool.
   * Deterministic from `t`, so it needs no particle state of its own.
   * opts: { groundY, W, H, lanterns, dust, glow, vignette } */
  function ambient(ctx, t, opts) {
    opts = opts || {};
    const W = opts.W != null ? opts.W : ctx.canvas.width;
    const H = opts.H != null ? opts.H : ctx.canvas.height;
    const groundY = opts.groundY != null ? opts.groundY : H - 96;
    const ts = t / 1000;

    // (1) warm light glow from upper area — gives the "lantern-lit" mood
    if (opts.glow !== false) {
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      const breathe = 0.5 + 0.5 * Math.sin(ts * 0.7);
      const gy = H * 0.18;
      const glow = ctx.createRadialGradient(W * 0.5, gy, 10, W * 0.5, gy, H * 0.7);
      glow.addColorStop(0, rgba(PAL.ember, 0.05 + 0.025 * breathe));
      glow.addColorStop(1, rgba(PAL.ember, 0));
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, W, H);
      ctx.restore();
    }

    // (2) floating dust / spore motes drifting in the light shaft
    if (opts.dust !== false) {
      ctx.save();
      const N = opts.dust === true || opts.dust == null ? 34 : opts.dust;
      ctx.globalCompositeOperation = "lighter";
      for (let i = 0; i < N; i++) {
        // each mote has a stable pseudo-random identity from i
        const seed = i * 12.9898;
        const baseX = frac(Math.sin(seed) * 43758.5453) * W;
        const baseY = frac(Math.sin(seed + 1) * 43758.5453);
        const sp = 6 + frac(Math.sin(seed + 2) * 7919) * 14;     // px/s upward
        const sway = 18 + frac(Math.sin(seed + 3) * 5273) * 30;
        const yy = (baseY * (groundY + 60) - ts * sp);
        const y = mod(yy, groundY + 60);
        const x = baseX + Math.sin(ts * 0.5 + seed) * sway;
        const tw = 0.25 + 0.35 * (0.5 + 0.5 * Math.sin(ts * 2 + seed * 3));
        const r = 0.6 + frac(Math.sin(seed + 4) * 1000) * 1.4;
        ctx.fillStyle = rgba(PAL.ember, tw * 0.5);
        ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
      }
      ctx.restore();
    }

    // (3) hanging paper lanterns that sway (pendulum) on their cords
    if (opts.lanterns !== false) {
      const xs = opts.lanterns && opts.lanterns.length ? opts.lanterns
        : [{ x: W * 0.30, top: 16, r: 22 }, { x: W * 0.70, top: 16, r: 22 }];
      for (let i = 0; i < xs.length; i++) hangingLantern(ctx, xs[i], ts, i);
    }

    // (4) vignette — drawn last so it sits over everything (readability)
    if (opts.vignette !== false) {
      const vg = ctx.createRadialGradient(W / 2, H * 0.42, H * 0.18, W / 2, H * 0.52, H * 0.95);
      vg.addColorStop(0, "rgba(0,0,0,0)");
      vg.addColorStop(0.7, "rgba(0,0,0,0.10)");
      vg.addColorStop(1, "rgba(8,6,4,0.46)");
      ctx.fillStyle = vg;
      ctx.fillRect(0, 0, W, H);
    }
  }

  // one swaying paper lantern hung from the top beam
  function hangingLantern(ctx, L, ts, idx) {
    const cordTop = L.top || 16;
    const cordLen = 26 + L.r;
    // pendulum: gentle, phase-offset per lantern
    const ang = Math.sin(ts * 0.9 + idx * 1.7) * 0.10;
    const px = L.x + Math.sin(ang) * cordLen;
    const py = cordTop + Math.cos(ang) * cordLen;
    const r = L.r;

    ctx.save();
    // cord
    ctx.strokeStyle = "rgba(30,20,12,0.8)";
    ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(L.x, cordTop - 6); ctx.lineTo(px, py - r); ctx.stroke();

    // cast glow first (under the body)
    ctx.globalCompositeOperation = "lighter";
    const flick = 0.85 + 0.15 * Math.sin(ts * 7 + idx * 2.3) * Math.sin(ts * 2.1 + idx);
    const glow = ctx.createRadialGradient(px, py, 2, px, py, r * 3.2);
    glow.addColorStop(0, rgba(PAL.ember, 0.30 * flick));
    glow.addColorStop(0.5, rgba(PAL.orange, 0.10 * flick));
    glow.addColorStop(1, rgba(PAL.orange, 0));
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(px, py, r * 3.2, 0, TAU); ctx.fill();
    ctx.globalCompositeOperation = "source-over";

    // body — lacquer-red lantern with ribs
    ctx.translate(px, py);
    ctx.rotate(ang * 0.5);
    const body = ctx.createLinearGradient(-r, 0, r, 0);
    body.addColorStop(0, "rgba(90,24,18,1)");
    body.addColorStop(0.5, rgba(PAL.lacquer, 1));
    body.addColorStop(1, "rgba(70,18,14,1)");
    ctx.fillStyle = body;
    ctx.beginPath(); ctx.ellipse(0, 0, r, r * 1.18, 0, 0, TAU); ctx.fill();
    // warm interior bleed
    ctx.globalCompositeOperation = "lighter";
    ctx.fillStyle = rgba(PAL.ember, 0.22 * flick);
    ctx.beginPath(); ctx.ellipse(0, 0, r * 0.7, r * 0.85, 0, 0, TAU); ctx.fill();
    ctx.globalCompositeOperation = "source-over";
    // ribs + caps
    ctx.strokeStyle = "rgba(20,12,8,0.5)";
    ctx.lineWidth = 1;
    for (let k = -2; k <= 2; k++) {
      ctx.beginPath();
      ctx.ellipse(0, 0, Math.abs(k) * r * 0.33 + 1, r * 1.18, 0, 0, TAU);
      ctx.stroke();
    }
    ctx.fillStyle = "rgba(20,14,9,0.95)";
    ctx.fillRect(-r * 0.35, -r * 1.24, r * 0.7, r * 0.16);
    ctx.fillRect(-r * 0.35, r * 1.08, r * 0.7, r * 0.16);
    ctx.restore();
  }

  function frac(x) { return x - Math.floor(x); }
  function mod(a, n) { return ((a % n) + n) % n; }

  // ===================================================================
  //  Stateful instance — owns particle pools, shake, stinger clock.
  // ===================================================================
  function create(cfg) {
    cfg = cfg || {};
    const impacts = [];     // { x, y, born, life, dir, color, seed, scale }
    const streaks = [];     // { x, y, vx, vy, born, life, color, w }
    const puffs = [];       // { x, y, vx, vy, born, life, r0, r1 }
    let shakeMag = 0;       // current shake amplitude (px), decays over time
    let shakeSeed = Math.random() * 1000;
    let stinger = null;     // { born, dur, winner, color }
    let last = 0;           // last tick timestamp
    let clock = 0;          // monotonically increasing ms

    const MAX = cfg.maxParticles || 260;

    function tick(now) {
      if (!last) last = now;
      const dt = Math.min(64, now - last);   // clamp big gaps (tab switch)
      last = now;
      clock = now;
      // shake decays exponentially toward 0
      shakeMag *= Math.pow(0.0001, dt / 1000);   // ~halves every ~75ms
      if (shakeMag < 0.05) shakeMag = 0;
      return dt;
    }

    // ---- triggers --------------------------------------------------
    function spawnImpact(x, y, opts) {
      opts = opts || {};
      impacts.push({
        x, y, born: clock,
        life: opts.life || 340,
        dir: opts.facing || opts.dir || 0,
        color: opts.color || PAL.orange,
        seed: Math.random() * TAU,
        scale: opts.scale || 1,
      });
      // a punch landing also kicks a little shake + a few forward sparks
      shake((opts.shake != null ? opts.shake : 5) * (opts.scale || 1));
      const dir = opts.facing || opts.dir || 1;
      const n = opts.sparks != null ? opts.sparks : 7;
      for (let i = 0; i < n; i++) {
        const a = rand(-0.7, 0.7) + (dir >= 0 ? 0 : Math.PI);
        const sp = rand(120, 320);
        streaks.push({
          x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - rand(20, 90),
          born: clock, life: rand(180, 360),
          color: i % 3 === 0 ? PAL.ember : opts.color || PAL.orange,
          w: rand(1.4, 3),
        });
      }
      trim();
    }

    // big launch: heavy shake + a fan of motion streaks trailing the fighter
    function knockback(x, y, opts) {
      opts = opts || {};
      const dir = opts.dir || 1;
      shake(opts.intensity != null ? opts.intensity : 14);
      const n = opts.streaks != null ? opts.streaks : 14;
      for (let i = 0; i < n; i++) {
        const spread = rand(-0.45, 0.45);
        const a = Math.atan2(-0.15, dir) + spread;
        const sp = rand(260, 620);
        streaks.push({
          x: x - dir * rand(0, 18), y: y - rand(0, 30),
          vx: Math.cos(a) * sp, vy: Math.sin(a) * sp,
          born: clock, life: rand(220, 480),
          color: i % 4 === 0 ? PAL.ink : opts.color || PAL.orange,
          w: rand(2, 5),
        });
      }
      // a dust kick at the takeoff point
      spawnDust(x, opts.groundY != null ? opts.groundY : y, { dir, power: 1.3 });
      trim();
    }

    // landing / dash puff — low, wide, settles outward along the ground
    function spawnDust(x, groundY, opts) {
      opts = opts || {};
      const dir = opts.dir || 0;
      const power = opts.power || 1;
      const n = Math.round((opts.count || 9) * power);
      for (let i = 0; i < n; i++) {
        const side = dir !== 0 ? dir * (i % 2 ? 1 : 0.4) : (i % 2 ? 1 : -1);
        const sp = rand(30, 110) * power;
        puffs.push({
          x: x + rand(-6, 6),
          y: groundY - rand(0, 4),
          vx: side * sp + rand(-15, 15),
          vy: -rand(8, 46) * power,
          born: clock, life: rand(360, 720),
          r0: rand(2, 5), r1: rand(10, 22) * power,
        });
      }
      trim();
    }

    function ringOut(winner, opts) {
      opts = opts || {};
      stinger = {
        born: clock,
        dur: opts.dur || 2600,
        winner: winner || null,
        color: opts.color || PAL.orange,
      };
      shake(opts.intensity != null ? opts.intensity : 22);
    }

    function shake(intensity) { shakeMag = Math.max(shakeMag, intensity); }

    // host wraps its scene draw with this; returns the applied offset
    function applyShake(ctx) {
      if (shakeMag <= 0) return { x: 0, y: 0 };
      // two-octave noise so it feels like a hit, not a sine wobble
      const tt = clock / 1000;
      const ox = (Math.sin(tt * 57 + shakeSeed) + 0.5 * Math.sin(tt * 113)) * shakeMag;
      const oy = (Math.cos(tt * 61 + shakeSeed) + 0.5 * Math.cos(tt * 127)) * shakeMag * 0.7;
      ctx.translate(ox, oy);
      return { x: ox, y: oy };
    }

    // ---- world-space particle render (call inside the shaken transform)
    function drawWorld(ctx, now) {
      const t = now != null ? now : clock;

      // streaks first (behind impacts)
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      ctx.lineCap = "round";
      for (let i = streaks.length - 1; i >= 0; i--) {
        const s = streaks[i];
        const p = (t - s.born) / s.life;
        if (p >= 1) { streaks.splice(i, 1); continue; }
        const fade = 1 - p;
        const len = 0.04;                              // trail = last 40ms of travel
        const tx = s.x + s.vx * (p) * (s.life / 1000);
        const ty = s.y + s.vy * (p) * (s.life / 1000) + 0.5 * 180 * Math.pow(p * s.life / 1000, 2);
        const bx = s.x + s.vx * Math.max(0, p - len) * (s.life / 1000);
        const by = s.y + s.vy * Math.max(0, p - len) * (s.life / 1000);
        ctx.strokeStyle = rgba(s.color, 0.7 * fade);
        ctx.lineWidth = s.w * fade;
        ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(tx, ty); ctx.stroke();
      }
      ctx.restore();

      // dust puffs (normal blend — they're smoke, not light)
      ctx.save();
      for (let i = puffs.length - 1; i >= 0; i--) {
        const d = puffs[i];
        const p = (t - d.born) / d.life;
        if (p >= 1) { puffs.splice(i, 1); continue; }
        const e = easeOut(p);
        const ageS = (t - d.born) / 1000;
        const px = d.x + d.vx * ageS;
        const py = d.y + d.vy * ageS + 60 * ageS * ageS;  // slight settle
        const r = lerp(d.r0, d.r1, e);
        const a = (1 - p) * 0.22;
        const g = ctx.createRadialGradient(px, py, 0, px, py, r);
        g.addColorStop(0, rgba(PAL.smoke, a));
        g.addColorStop(1, rgba(PAL.smoke, 0));
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(px, py, r, 0, TAU); ctx.fill();
      }
      ctx.restore();

      // impacts on top
      for (let i = impacts.length - 1; i >= 0; i--) {
        const m = impacts[i];
        const p = (t - m.born) / m.life;
        if (p >= 1) { impacts.splice(i, 1); continue; }
        drawImpact(ctx, m.x, m.y, p, { dir: m.dir, color: m.color, seed: m.seed, scale: m.scale });
      }
    }

    // ---- screen-space hud (stinger) -------------------------------
    function drawHud(ctx, now) {
      const t = now != null ? now : clock;
      if (!stinger) return;
      const p = (t - stinger.born) / stinger.dur;
      if (p >= 1) { stinger = null; return; }
      ringOutStinger(ctx, stinger.winner, p, { color: stinger.color });
    }

    // ambient delegates to the stateless pass but uses the live clock
    function ambientPass(ctx, now, opts) {
      ambient(ctx, now != null ? now : clock, opts);
    }

    function trim() {
      // hard ceiling: drop oldest streaks → puffs → impacts (impacts last,
      // they're rare + visually important) until total <= MAX.
      let over = streaks.length + puffs.length + impacts.length - MAX;
      if (over <= 0) return;
      const cut = (arr) => {
        if (over <= 0) return;
        const n = Math.min(arr.length, over);
        arr.splice(0, n);
        over -= n;
      };
      cut(streaks); cut(puffs); cut(impacts);
    }

    function clear() { impacts.length = streaks.length = puffs.length = 0; stinger = null; shakeMag = 0; }
    function isShaking() { return shakeMag > 0; }
    function hasStinger() { return !!stinger; }

    return {
      tick, applyShake, isShaking,
      spawnImpact, knockback, spawnDust, ringOut, shake,
      drawWorld, drawHud, ambient: ambientPass, hasStinger,
      clear,
      get counts() { return { impacts: impacts.length, streaks: streaks.length, puffs: puffs.length }; },
    };
  }

  // ----- module surface ----------------------------------------------
  let _default = null;
  function defaultInstance() { return _default || (_default = create()); }

  const API = {
    create,
    PALETTE: PAL,
    // stateless primitives
    drawImpact,
    spawnImpact: drawImpact,          // alias matching the brief's signature
    ringOutStinger,
    ambient,
    // convenience that drive a shared default instance
    shake: (i) => defaultInstance().shake(i),
    _default: defaultInstance,
  };

  root.DojoFX = API;
  if (typeof module !== "undefined" && module.exports) module.exports = API;
})(typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : typeof global !== "undefined" ? global : this);
