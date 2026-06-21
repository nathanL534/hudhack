/* =============================================================================
 * sprite_anim.js — reusable sprite-animation engine for the DOJO replay viewer
 * -----------------------------------------------------------------------------
 * Animates a 2D platform-fighter by playing the right clip (idle / walk / jump /
 * punch / hit / fall / KO) based on per-frame replay state. Two render paths
 * behind ONE interface:
 *
 *   1. Real sprite sheets (assets/sprites/<char>/<Clip>.png) — horizontal strips
 *      of fixed-size frames, the classic "Fighting Game in JS" layout.
 *   2. A clean PROCEDURAL karate-gi figure (vector/canvas) drawn with the SAME
 *      interface, so the viewer is alive RIGHT NOW and just gets crisper when
 *      real art lands. The fallback is the default and is fully animated.
 *
 * No build step. Works as an ES module (`import { SpriteAnimator } from ...`)
 * AND as a classic <script> (exposes `window.SpriteAnim`).
 *
 * ----------------------------------------------------------------------------
 * QUICK USE (procedural fallback — works with zero assets):
 *
 *   const anim = SpriteAnimator.procedural({ team: "trained" }); // or "base"
 *   // per render frame:
 *   const st = SpriteAnimator.selectAction(curFrame.p1, prevFrame?.p1);
 *   anim.draw(ctx, st.action, st.facing, screenX, groundScreenY, scale, nowMs);
 *
 * QUICK USE (real sheets, classic samuraiMack/kenji layout):
 *
 *   const anim = SpriteAnimator.fromCharacter("samuraiMack", { team: "base" });
 *   await anim.ready();                 // optional; draw() degrades gracefully
 *   anim.draw(ctx, "walk", -1, x, y, scale, nowMs);
 *
 * The (x, y) passed to draw() is the SCREEN-SPACE point where the fighter's FEET
 * stand (ground contact). The viewer already owns world->screen; this module is
 * deliberately screen-space so it drops straight into viewer.html's render loop.
 * =========================================================================== */

(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api; // node/CJS
  root.SpriteAnim = api;                                                  // <script>
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  // ------------------------------------------------------------------ palette
  // Mirrors viewer.html's dojo palette so the fallback figure is visually
  // continuous with the rest of the page. Kept local so this file is standalone.
  const PALETTE = {
    trained: "#c4453c", // muted red  — Player 2 / Trained student
    base:    "#4a63b0", // muted blue — Player 1 / Base student
    robe:    "#f3ecda",
    robeHi:  "#fbf6ec",
    robeLo:  "#d9cdb0",
    robeLine:"#b9ad8d",
    skin:    "#f0cda0",
    skinLo:  "#d9b184",
    hair:    "#2a1d14",
    ink:     "#33281d",
    sweat:   "rgba(224,140,90,0.9)",
  };

  // The canonical animation states this engine understands.
  const ACTIONS = ["idle", "walk", "jump", "fall", "punch", "hit", "ko"];

  // ----------------------------------------------------------------- helpers
  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  function hexToRgb(hex) {
    const m = /^#?([0-9a-f]{6})$/i.exec(String(hex).trim());
    if (!m) return null;
    const n = parseInt(m[1], 16);
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
  }
  // viewer.html's shade(): additive lighten/darken, returns rgb() string.
  function shade(hex, amt) {
    const c = hexToRgb(hex);
    if (!c) return hex;
    const f = (v) => clamp(Math.round(v + amt), 0, 255);
    return `rgb(${f(c.r)},${f(c.g)},${f(c.b)})`;
  }
  function withAlpha(hex, a) {
    const c = hexToRgb(hex) || { r: 200, g: 160, b: 80 };
    return `rgba(${c.r},${c.g},${c.b},${a})`;
  }

  function teamColor(team) {
    if (team === "trained" || team === "p2" || team === "red") return PALETTE.trained;
    if (team === "base" || team === "p1" || team === "blue") return PALETTE.base;
    // allow a raw hex too
    return hexToRgb(team) ? String(team) : PALETTE.base;
  }

  // ===========================================================================
  // ACTION SELECTION
  // ---------------------------------------------------------------------------
  // Map a replay fighter's per-frame state -> an animation state + facing.
  //
  // Replay schema (observed across replays/*.json):
  //   fighter = { x, y, facing: 1|-1, action: "idle"|"left"|"right"|"jump"|"punch" }
  //   y is in WORLD units; ground is y==0 (platform.y). No vx/vy in the schema.
  //
  // We therefore derive motion from `action` + `y` + frame deltas:
  //   - punch action this frame                  -> "punch"
  //   - knocked back (big +y or +x jump vs prev,
  //     and not self-initiated)                  -> "hit"
  //   - airborne (y above ground) and rising/at  -> "jump"
  //   - airborne and descending                  -> "fall"
  //   - action left/right (or x moved on ground) -> "walk"
  //   - else                                     -> "idle"
  //
  // `prev` is optional but makes hit/fall detection possible. `opts` lets the
  // viewer feed richer signals later (explicit gotHit / vy) without changing
  // call sites.
  // ===========================================================================
  function selectAction(cur, prev, opts) {
    opts = opts || {};
    if (!cur) return { action: "idle", facing: 1 };

    const facing = cur.facing === -1 ? -1 : (cur.facing === 1 ? 1 : (opts.facing || 1));
    const a = String(cur.action || "idle").toLowerCase();
    const groundY = opts.groundY != null ? opts.groundY : 0;
    const airThresh = opts.airThreshold != null ? opts.airThreshold : 0.05;

    const y = num(cur.y);
    const py = prev ? num(prev.y) : y;
    const px = prev ? num(prev.x) : num(cur.x);
    const dy = y - py;
    const dx = num(cur.x) - px;

    const airborne = (y - groundY) > airThresh;

    // 1) explicit KO / ringout (viewer can pass opts.ko or action "ko"/"death")
    if (opts.ko || a === "ko" || a === "death" || a === "dead") {
      return { action: "ko", facing };
    }

    // 2) hit / knockback. No damage event in schema, so infer from kinematics:
    //    a sudden launch (large +dy) or shove (large |dx|) the fighter did NOT
    //    initiate. Viewer may override with an explicit opts.gotHit.
    const hitThresh = opts.hitThreshold != null ? opts.hitThreshold : 0.6;
    const selfMoved = a === "left" || a === "right" || a === "jump";
    const launched = dy > hitThresh || (Math.abs(dx) > hitThresh && !selfMoved);
    if (opts.gotHit || launched) {
      return { action: "hit", facing };
    }

    // 3) attack
    if (a === "punch" || a === "attack" || opts.attacking) {
      return { action: "punch", facing };
    }

    // 4) airborne -> jump (rising / apex) vs fall (descending)
    const vy = opts.vy != null ? opts.vy : dy;
    if (airborne || a === "jump") {
      return { action: vy < -airThresh ? "fall" : "jump", facing };
    }

    // 5) horizontal locomotion
    if (a === "left" || a === "right" || a === "walk" || a === "run" ||
        Math.abs(dx) > (opts.walkThreshold != null ? opts.walkThreshold : 0.01)) {
      return { action: "walk", facing };
    }

    // 6) default
    return { action: "idle", facing };
  }
  function num(v) { return typeof v === "number" && isFinite(v) ? v : 0; }

  // ===========================================================================
  // CLIP: a single animation (one row / one strip). Decoupled from source so
  // procedural and sheet paths share the cadence math.
  // ===========================================================================
  class Clip {
    constructor({ name, frames, fps, loop = true, image = null, frameW = 0, frameH = 0, row = 0, offsetX = 0 }) {
      this.name = name;
      this.frames = Math.max(1, frames | 0);
      this.fps = fps > 0 ? fps : 10;
      this.loop = loop;
      this.image = image;     // HTMLImageElement or null (procedural)
      this.frameW = frameW;
      this.frameH = frameH;
      this.row = row;         // source row index (multi-row sheets)
      this.offsetX = offsetX; // source column offset (multi-clip sheets)
    }
    // Which frame index to show at absolute time nowMs, given when this clip
    // started (startMs). Non-looping clips clamp on the last frame.
    frameAt(nowMs, startMs) {
      const elapsed = Math.max(0, nowMs - startMs);
      const raw = Math.floor((elapsed / 1000) * this.fps);
      if (this.loop) return raw % this.frames;
      return Math.min(raw, this.frames - 1);
    }
    done(nowMs, startMs) {
      if (this.loop) return false;
      const elapsed = Math.max(0, nowMs - startMs);
      return (elapsed / 1000) * this.fps >= this.frames;
    }
  }

  // ===========================================================================
  // SpriteAnimator — owns the per-character clip set + a render path, and a
  // small state machine so non-looping clips (punch/hit) play out cleanly even
  // though the caller passes a fresh action each render frame.
  // ===========================================================================
  class SpriteAnimator {
    /**
     * @param {object} cfg
     *   cfg.team     "trained"|"base"|"p1"|"p2" | hex   (headband/belt color)
     *   cfg.clips    { actionName: Clip }                (built by factories)
     *   cfg.procedural  boolean                          (force vector figure)
     *   cfg.baseHeight  px height of the figure at scale=1 (default 120)
     */
    constructor(cfg = {}) {
      this.team = cfg.team || "base";
      this.color = teamColor(this.team);
      this.clips = cfg.clips || {};
      this.procedural = cfg.procedural !== false ? !hasAnyImage(this.clips) : false;
      if (cfg.procedural === true) this.procedural = true;
      this.baseHeight = cfg.baseHeight || 120;

      // playback state machine
      this._cur = "idle";
      this._startMs = 0;
      this._lastNow = 0;

      // a deferred-ready promise so callers can await asset load if they want
      this._readyResolve = null;
      this._readyP = new Promise((res) => (this._readyResolve = res));
      this._pending = countPending(this.clips);
      if (this._pending === 0 && this._readyResolve) this._readyResolve(this);
    }

    /** Resolves once all declared sheet images have loaded (or immediately for procedural). */
    ready() { return this._readyP; }

    /** True if at least one real sheet is loaded and usable. */
    hasSprites() { return !this.procedural && hasAnyImage(this.clips); }

    _resolveClip(action) {
      // alias map so callers can pass replay-ish names too
      const a = (action || "idle").toLowerCase();
      if (this.clips[a]) return a;
      const alias = {
        run: "walk", left: "walk", right: "walk", move: "walk",
        attack: "punch", punch1: "punch", punch2: "punch",
        takehit: "hit", hurt: "hit", knockback: "hit",
        death: "ko", dead: "ko",
        rise: "jump", air: "jump", descend: "fall",
      };
      if (alias[a] && this.clips[alias[a]]) return alias[a];
      if (this.clips.idle) return "idle";
      return Object.keys(this.clips)[0] || "idle";
    }

    /**
     * Core entry point. Draw the fighter at (x,y) where (x,y) is the SCREEN
     * point under the figure's feet.
     *
     * @param {CanvasRenderingContext2D} ctx
     * @param {string} action  one of ACTIONS (or a replay alias)
     * @param {number} facing  1 (right) | -1 (left)
     * @param {number} x        screen x of feet
     * @param {number} y        screen y of feet (ground contact)
     * @param {number} scale    1 == baseHeight px tall
     * @param {number} nowMs    monotonic clock (performance.now()); defaults to now
     */
    draw(ctx, action, facing, x, y, scale, nowMs) {
      if (nowMs == null) nowMs = (typeof performance !== "undefined" ? performance.now() : Date.now());
      this._lastNow = nowMs;
      scale = scale || 1;
      facing = facing === -1 ? -1 : 1;

      const want = this._resolveClip(action);

      // State machine: restart the clip clock only on a real transition, so a
      // looping walk keeps cycling and a one-shot punch plays start->end. A
      // non-looping clip that is still playing "holds" priority over idle/walk.
      const curClip = this.clips[this._cur];
      const oneShotBusy =
        curClip && !curClip.loop && !curClip.done(nowMs, this._startMs) &&
        (this._cur === "punch" || this._cur === "hit");

      if (want !== this._cur) {
        const overridePriority = want === "hit" || want === "ko"; // damage interrupts
        if (!oneShotBusy || overridePriority) {
          this._cur = want;
          this._startMs = nowMs;
        }
      }
      const clipName = this._cur;
      const clip = this.clips[clipName];
      if (!clip) return;

      const frameIdx = clip.frameAt(nowMs, this._startMs);

      ctx.save();
      // flip around the feet point for left-facing
      ctx.translate(x, y);
      if (facing === -1) ctx.scale(-1, 1);

      if (clip.image && clip.image.complete && clip.image.naturalWidth > 0) {
        this._drawSheetFrame(ctx, clip, frameIdx, scale);
      } else {
        drawProceduralFighter(ctx, {
          action: clipName, frame: frameIdx, frames: clip.frames,
          color: this.color, height: this.baseHeight * scale,
          nowMs,
        });
      }
      ctx.restore();
    }

    _drawSheetFrame(ctx, clip, frameIdx, scale) {
      const img = clip.image;
      const fw = clip.frameW || img.naturalHeight; // square frames if unspecified
      const fh = clip.frameH || img.naturalHeight;
      const sx = clip.offsetX + frameIdx * fw;
      const sy = clip.row * fh;
      // Scale so the sheet's frame height maps to baseHeight*scale, feet at (0,0).
      const drawH = this.baseHeight * scale;
      const drawW = drawH * (fw / fh);
      // Classic fighting-game sheets pad the figure inside the frame; we anchor
      // feet to the bottom of the frame and center horizontally.
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, sx, sy, fw, fh, -drawW / 2, -drawH, drawW, drawH);
    }

    // ---------------------------------------------------------------- factories
    /** Pure procedural fighter — no assets required. */
    static procedural(cfg = {}) {
      return new SpriteAnimator({
        team: cfg.team || "base",
        procedural: true,
        baseHeight: cfg.baseHeight || 120,
        clips: buildProceduralClips(cfg.fps || {}),
      });
    }

    /**
     * Build from an explicit manifest:
     *   { frameW, frameH, baseHeight, fps, clips: { walk: {src, frames, fps, loop}, ... } }
     * Missing/failed images fall back to procedural per-clip automatically.
     */
    static fromManifest(manifest, cfg = {}) {
      const clips = {};
      const fw = manifest.frameW || 0, fh = manifest.frameH || 0;
      for (const name of Object.keys(manifest.clips || {})) {
        const c = manifest.clips[name];
        const image = c.src ? loadImage(resolveSrc(c.src, manifest.basePath)) : null;
        clips[name] = new Clip({
          name,
          frames: c.frames || 1,
          fps: c.fps || manifest.fps || 10,
          loop: c.loop != null ? c.loop : !ONE_SHOT[name],
          image,
          frameW: c.frameW || fw,
          frameH: c.frameH || fh,
          row: c.row || 0,
          offsetX: c.offsetX || 0,
        });
      }
      // ensure procedural coverage for any canonical action the manifest omits
      ensureCoverage(clips);
      return new SpriteAnimator({
        team: cfg.team || manifest.team || "base",
        baseHeight: cfg.baseHeight || manifest.baseHeight || 120,
        clips,
      });
    }

    /**
     * Convenience for the assets shipped in this repo:
     *   assets/sprites/<char>/{Idle,Run,Jump,Fall,Attack1,TakeHit,Death}.png
     * Each is a horizontal strip of 200x200 frames. Frame counts differ per
     * character; we pass the known-good counts for samuraiMack / kenji and a
     * sane default otherwise.
     */
    static fromCharacter(charName, cfg = {}) {
      const basePath = (cfg.basePath || "assets/sprites/") + charName + "/";
      const counts = CHAR_FRAMES[charName] || CHAR_FRAMES._default;
      const F = 200; // frame size in these sheets
      const clip = (file, frames, fps, loop, action) => ({
        action,
        src: basePath + file,
        frames, fps, loop, frameW: F, frameH: F,
      });
      const manifestClips = {
        idle:  clip("Idle.png",     counts.idle,  8,  true,  "idle"),
        walk:  clip("Run.png",      counts.run,   12, true,  "walk"),
        jump:  clip("Jump.png",     counts.jump,  8,  false, "jump"),
        fall:  clip("Fall.png",     counts.fall,  8,  false, "fall"),
        punch: clip("Attack1.png",  counts.attack,14, false, "punch"),
        hit:   clip("TakeHit.png",  counts.hit,   10, false, "hit"),
        ko:    clip("Death.png",    counts.death, 8,  false, "ko"),
      };
      return SpriteAnimator.fromManifest(
        { frameW: F, frameH: F, fps: 10, clips: indexByAction(manifestClips) },
        cfg
      );
    }
  }

  // Known frame counts for the bundled sheets (verified from PNG widths / 200).
  const CHAR_FRAMES = {
    samuraiMack: { idle: 8, run: 8, jump: 2, fall: 2, attack: 6, hit: 4, death: 6 },
    kenji:       { idle: 4, run: 8, jump: 2, fall: 2, attack: 4, hit: 3, death: 7 },
    _default:    { idle: 4, run: 6, jump: 2, fall: 2, attack: 4, hit: 3, death: 5 },
  };

  const ONE_SHOT = { jump: true, fall: true, punch: true, hit: true, ko: true };

  // ----------------------------------------------------------- manifest plumbing
  function indexByAction(m) {
    const out = {};
    for (const k of Object.keys(m)) {
      const { action, ...rest } = m[k];
      out[action || k] = rest;
    }
    return out;
  }
  function resolveSrc(src, basePath) {
    if (!basePath || /^(https?:)?\/\//.test(src) || src.startsWith("/")) return src;
    return basePath + src;
  }
  function loadImage(src) {
    if (typeof Image === "undefined") return null; // node / headless: stay procedural
    const img = new Image();
    img.src = src;
    return img;
  }
  function hasAnyImage(clips) {
    for (const k in clips) {
      const im = clips[k] && clips[k].image;
      if (im && (im.complete ? im.naturalWidth > 0 : true)) return true;
    }
    return false;
  }
  function countPending(clips) {
    let n = 0;
    for (const k in clips) {
      const im = clips[k] && clips[k].image;
      if (im && !im.complete) {
        n++;
        const done = () => {};
        im.addEventListener && im.addEventListener("load", done);
        im.addEventListener && im.addEventListener("error", done);
      }
    }
    return n;
  }
  function ensureCoverage(clips) {
    const proc = buildProceduralClips({});
    for (const a of ACTIONS) if (!clips[a]) clips[a] = proc[a];
  }

  // ===========================================================================
  // PROCEDURAL FALLBACK
  // ---------------------------------------------------------------------------
  // A crisp canvas karate figure: white gi, team-colored headband + belt, drawn
  // in the same chibi-leaning language as viewer.html's avatar. Every action is
  // animated by computing joint angles from (frame / frames) so it cycles
  // smoothly and reads clearly even at small scale.
  //
  // Coordinate convention inside drawProceduralFighter: origin (0,0) is at the
  // FEET, +x right, +y down is screen-down — but we build UP from the feet using
  // NEGATIVE y for height. height = full figure height in px.
  // ===========================================================================
  function buildProceduralClips(fps) {
    const f = (n, d) => (fps && fps[n]) || d;
    return {
      idle:  new Clip({ name: "idle",  frames: 8,  fps: f("idle", 6),  loop: true }),
      walk:  new Clip({ name: "walk",  frames: 8,  fps: f("walk", 12), loop: true }),
      jump:  new Clip({ name: "jump",  frames: 4,  fps: f("jump", 10), loop: false }),
      fall:  new Clip({ name: "fall",  frames: 4,  fps: f("fall", 10), loop: false }),
      punch: new Clip({ name: "punch", frames: 6,  fps: f("punch",16), loop: false }),
      hit:   new Clip({ name: "hit",   frames: 5,  fps: f("hit",  12), loop: false }),
      ko:    new Clip({ name: "ko",    frames: 6,  fps: f("ko",   8),  loop: false }),
    };
  }

  function drawProceduralFighter(ctx, o) {
    const H = o.height;
    const color = o.color;
    const t = o.frames > 1 ? o.frame / (o.frames - 1) : 0; // 0..1 progress
    const loopT = o.frame / Math.max(1, o.frames);          // 0..1 cyclic phase
    const P = PALETTE;

    // ---- skeleton metrics (proportions relative to total height H) ----
    const hipY    = -H * 0.46;   // pelvis
    const shY     = -H * 0.74;   // shoulders
    const neckY   = -H * 0.80;
    const headR   = H * 0.115;
    const headCY  = -H * 0.88;
    const legLen  = -hipY;       // hips to floor
    const armLen  = H * 0.30;
    const bodyW   = H * 0.18;

    // crouch / vertical bob offset per action (raises or lowers the whole rig)
    let bob = 0, lean = 0;
    let lArm = -0.5, rArm = 0.4;     // shoulder angles (radians; 0 = straight down)
    let lLeg = 0.16, rLeg = -0.16;   // hip angles
    let punchEx = 0;                  // 0..1 lead-arm extension
    let headTilt = 0;
    let kneeBend = 0.12;

    switch (o.action) {
      case "idle": {
        // gentle breathing bob + small arm sway
        const s = Math.sin(loopT * Math.PI * 2);
        bob = s * H * 0.012;
        lArm = -0.42 + s * 0.06;
        rArm =  0.42 - s * 0.06;
        headTilt = s * 0.04;
        break;
      }
      case "walk": {
        // contralateral arm/leg swing, body rise-fall twice per cycle
        const ph = loopT * Math.PI * 2;
        lLeg =  Math.sin(ph) * 0.55;
        rLeg = -Math.sin(ph) * 0.55;
        lArm = -0.5 - Math.sin(ph) * 0.55;
        rArm =  0.5 - Math.sin(ph) * 0.55;
        bob  = -Math.abs(Math.cos(ph)) * H * 0.03; // up on midstep
        lean = 0.05;
        kneeBend = 0.28;
        break;
      }
      case "jump": {
        // tuck: knees up, arms up, slight backward lean
        kneeBend = 0.5 + t * 0.2;
        lLeg = 0.5; rLeg = 0.6;
        lArm = -1.5 + t * 0.3; rArm = -1.4 + t * 0.3;
        bob = -H * 0.02 * (1 - t);
        lean = -0.04;
        break;
      }
      case "fall": {
        // arms out for balance, legs reaching down
        lLeg = -0.25; rLeg = 0.25;
        lArm = -1.9; rArm = 1.9;
        kneeBend = 0.18;
        lean = 0.06;
        break;
      }
      case "punch": {
        // wind-up (0->0.3) then snap lead straight out (0.3->0.7) then recover
        const ph = clamp((t - 0.15) / 0.5, 0, 1);
        punchEx = ph < 1 ? easeOutBack(ph) : 1 - (t - 0.65) / 0.35;
        punchEx = clamp(punchEx, 0, 1);
        rArm = 0.2 - punchEx * 0.2;   // rear arm chambers
        lArm = -0.2;                  // lead arm handled via punchEx below
        lean = 0.10 + punchEx * 0.10;
        rLeg = -0.32; lLeg = 0.30;    // bracing stance
        kneeBend = 0.34;
        break;
      }
      case "hit": {
        // recoil: snap back, head whips, arms fly up, knees buckle
        const recoil = Math.sin(clamp(t, 0, 1) * Math.PI); // 0..1..0
        lean = -0.28 * recoil;
        bob = H * 0.02 * recoil;
        headTilt = -0.4 * recoil;
        lArm = -1.4 * recoil - 0.3;
        rArm =  1.4 * recoil + 0.3;
        lLeg = 0.3 * recoil; rLeg = -0.3 * recoil;
        kneeBend = 0.3 + 0.2 * recoil;
        break;
      }
      case "ko": {
        // collapse: rotate the whole rig toward the floor as t->1
        ctx.rotate(-t * (Math.PI / 2 - 0.12));
        ctx.translate(0, t * H * 0.04);
        lArm = -1.0; rArm = 1.0; lLeg = 0.2; rLeg = -0.2;
        headTilt = 0.3;
        break;
      }
    }

    ctx.save();
    ctx.translate(0, bob);
    ctx.rotate(lean);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    const ink = P.ink;
    const robe = P.robe;
    const limbW = Math.max(2, H * 0.085);

    // soft contact shadow under the feet
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.28)";
    ctx.beginPath();
    ctx.ellipse(0, 2, H * 0.16, H * 0.035, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // ---- helper to draw a 2-segment limb from a pivot ----
    const limb = (px, py, ang, len, bend, w, col) => {
      const midX = px + Math.sin(ang) * len * 0.5;
      const midY = py + Math.cos(ang) * len * 0.5;
      const endX = px + Math.sin(ang) * len * (0.5) + Math.sin(ang + bend) * len * 0.5;
      const endY = py + Math.cos(ang) * len * (0.5) + Math.cos(ang + bend) * len * 0.5;
      ctx.strokeStyle = col;
      ctx.lineWidth = w;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(midX, midY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
      return { x: endX, y: endY };
    };

    // ===== legs (behind torso) =====
    const hipX = 0;
    limb(hipX - bodyW * 0.18, hipY, rLeg, legLen, kneeBend, limbW, shade(robe, -34));
    limb(hipX + bodyW * 0.18, hipY, lLeg, legLen, kneeBend, limbW, shade(robe, -18));

    // ===== rear arm (behind torso) =====
    limb(bodyW * 0.34, shY, rArm, armLen, 0.35, limbW * 0.92, shade(robe, -28));

    // ===== torso / gi =====
    ctx.fillStyle = robe;
    ctx.strokeStyle = ink;
    ctx.lineWidth = Math.max(1.4, H * 0.012);
    ctx.beginPath();
    ctx.moveTo(-bodyW * 0.62, hipY + H * 0.02);
    ctx.lineTo(-bodyW * 0.78, shY);
    ctx.quadraticCurveTo(-bodyW * 0.5, shY - H * 0.02, 0, shY - H * 0.018);
    ctx.quadraticCurveTo(bodyW * 0.5, shY - H * 0.02, bodyW * 0.78, shY);
    ctx.lineTo(bodyW * 0.62, hipY + H * 0.02);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    // lapel V
    ctx.strokeStyle = P.robeLine;
    ctx.lineWidth = Math.max(1.2, H * 0.011);
    ctx.beginPath();
    ctx.moveTo(-bodyW * 0.30, shY + H * 0.02);
    ctx.lineTo(0, hipY - H * 0.10);
    ctx.lineTo(bodyW * 0.30, shY + H * 0.02);
    ctx.stroke();
    // gi shading edge
    ctx.fillStyle = withAlpha(P.robeLo, 0.6);
    ctx.beginPath();
    ctx.moveTo(bodyW * 0.62, hipY + H * 0.02);
    ctx.lineTo(bodyW * 0.78, shY);
    ctx.quadraticCurveTo(bodyW * 0.4, shY, bodyW * 0.34, hipY);
    ctx.closePath();
    ctx.fill();

    // ===== belt (team color) =====
    const beltY = hipY - H * 0.02;
    ctx.fillStyle = color;
    ctx.fillRect(-bodyW * 0.66, beltY, bodyW * 1.32, H * 0.05);
    ctx.fillStyle = shade(color, -34);
    ctx.fillRect(-bodyW * 0.66, beltY + H * 0.035, bodyW * 1.32, H * 0.012);
    // belt knot + tails
    ctx.fillStyle = shade(color, -16);
    ctx.fillRect(-bodyW * 0.10, beltY, bodyW * 0.2, H * 0.075);

    // ===== lead arm (in front) — drives the punch =====
    {
      let ang = lArm;
      let len = armLen;
      let bend = 0.4;
      if (o.action === "punch") {
        // extend straight out front (toward +x because facing handled by caller)
        ang = (Math.PI / 2) - 0.05;     // horizontal
        len = armLen * (0.7 + punchEx * 0.7);
        bend = 0.5 - punchEx * 0.5;     // straighten on extension
      }
      const hand = limb(-bodyW * 0.34, shY, ang, len, bend, limbW * 0.92, shade(robe, -6));
      // fist
      ctx.fillStyle = P.skin;
      ctx.strokeStyle = ink;
      ctx.lineWidth = Math.max(1, H * 0.009);
      ctx.beginPath();
      ctx.arc(hand.x, hand.y, limbW * 0.62, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      // impact spark on full punch extension
      if (o.action === "punch" && punchEx > 0.82) {
        ctx.strokeStyle = withAlpha("#e08c3f", 0.9 * (punchEx));
        ctx.lineWidth = Math.max(1.5, H * 0.014);
        for (let i = 0; i < 5; i++) {
          const a2 = (i / 5) * Math.PI * 2;
          const r0 = limbW * 0.9, r1 = limbW * (1.5 + 0.5 * Math.sin(o.nowMs / 40 + i));
          ctx.beginPath();
          ctx.moveTo(hand.x + Math.cos(a2) * r0, hand.y + Math.sin(a2) * r0);
          ctx.lineTo(hand.x + Math.cos(a2) * r1, hand.y + Math.sin(a2) * r1);
          ctx.stroke();
        }
      }
    }

    // ===== head =====
    ctx.save();
    ctx.translate(0, headCY);
    ctx.rotate(headTilt);
    // face
    ctx.fillStyle = P.skin;
    ctx.strokeStyle = ink;
    ctx.lineWidth = Math.max(1.3, H * 0.011);
    ctx.beginPath();
    ctx.ellipse(0, 0, headR, headR * 1.04, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    // hair cap
    ctx.fillStyle = P.hair;
    ctx.beginPath();
    ctx.moveTo(-headR * 0.98, -headR * 0.16);
    ctx.quadraticCurveTo(-headR * 1.05, -headR * 1.05, 0, -headR * 1.12);
    ctx.quadraticCurveTo(headR * 1.05, -headR * 1.05, headR * 0.98, -headR * 0.16);
    ctx.quadraticCurveTo(headR * 0.4, -headR * 0.42, 0, -headR * 0.30);
    ctx.quadraticCurveTo(-headR * 0.4, -headR * 0.42, -headR * 0.98, -headR * 0.16);
    ctx.closePath();
    ctx.fill();
    // headband (team color) — clipped to head
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(0, 0, headR, headR * 1.04, 0, 0, Math.PI * 2);
    ctx.clip();
    ctx.fillStyle = color;
    ctx.fillRect(-headR - 2, -headR * 0.36, headR * 2 + 4, headR * 0.34);
    ctx.fillStyle = shade(color, -30);
    ctx.fillRect(-headR - 2, -headR * 0.06, headR * 2 + 4, headR * 0.06);
    ctx.restore();
    // headband knot tail (trails behind = -x side)
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(-headR * 0.9, -headR * 0.2);
    ctx.quadraticCurveTo(-headR * 1.6, -headR * 0.05, -headR * 1.8, headR * 0.18);
    ctx.lineTo(-headR * 1.45, headR * 0.2);
    ctx.quadraticCurveTo(-headR * 1.15, headR * 0.02, -headR * 0.85, headR * 0.02);
    ctx.closePath();
    ctx.fill();
    // determined brow + eyes (face +x = forward)
    ctx.strokeStyle = shade(P.hair, 20);
    ctx.lineWidth = Math.max(1.4, H * 0.013);
    ctx.beginPath();
    ctx.moveTo(headR * 0.12, headR * 0.04);
    ctx.lineTo(headR * 0.55, headR * 0.16);
    ctx.stroke();
    ctx.fillStyle = ink;
    ctx.beginPath();
    ctx.arc(headR * 0.38, headR * 0.30, headR * 0.10, 0, Math.PI * 2);
    ctx.fill();
    // ko = X eye + droop already via rotation
    if (o.action === "ko") {
      ctx.strokeStyle = ink;
      ctx.lineWidth = Math.max(1.4, H * 0.013);
      const ex = headR * 0.38, ey = headR * 0.28, s = headR * 0.13;
      ctx.beginPath(); ctx.moveTo(ex - s, ey - s); ctx.lineTo(ex + s, ey + s); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ex - s, ey + s); ctx.lineTo(ex + s, ey - s); ctx.stroke();
    }
    ctx.restore(); // head

    ctx.restore(); // bob/lean
  }

  function easeOutBack(x) {
    const c1 = 1.70158, c3 = c1 + 1;
    return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
  }

  // ---------------------------------------------------------------- public API
  return {
    SpriteAnimator,
    Clip,
    selectAction,
    ACTIONS,
    PALETTE,
    teamColor,
    // expose for the demo / debugging
    _internal: { drawProceduralFighter, buildProceduralClips, shade, withAlpha },
  };
});

/* ES-module bridge: `import { SpriteAnimator, ... } from "./sprite_anim.js"`.
   The IIFE above also assigned globalThis.SpriteAnim for <script> users. We
   re-export the same object's members here. Guarded so CJS/<script> ignore it. */
export const SpriteAnimator = (globalThis.SpriteAnim || {}).SpriteAnimator;
export const Clip = (globalThis.SpriteAnim || {}).Clip;
export const selectAction = (globalThis.SpriteAnim || {}).selectAction;
export const ACTIONS = (globalThis.SpriteAnim || {}).ACTIONS;
export const PALETTE = (globalThis.SpriteAnim || {}).PALETTE;
export const teamColor = (globalThis.SpriteAnim || {}).teamColor;
export default (globalThis.SpriteAnim || {});
