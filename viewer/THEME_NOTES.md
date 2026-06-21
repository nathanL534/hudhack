# Dojo Replay Viewer — Theme Notes

Goal: take the existing dark "Sensei vs Sensei" replay viewer from clean-but-flat to
premium, demo-ready, and alive — without changing the markup or the JS behavior.
`theme.css` is a drop-in stylesheet; `theme_preview.html` shows the upgraded look in
isolation with static placeholder data.

---

## References studied

I looked at five categories of polished UI that share DNA with a fighting-game replay
viewer: esports stat overlays, fighting-game victory screens, and pro media-player
scrubbers. The point of each entry is the *specific move* I'm borrowing, not a vibe.

1. **Street Fighter 6 — battle HUD + result screen**
   https://www.streetfighter.com/6/en-us/ ·
   reference frames: https://www.eventhubs.com/street-fighter-6/
   - Borrowed: the **winner side commands the frame** — one side gets warm, saturated
     emphasis (glow, brighter portrait ring, a "WINS" stamp) while the loser desaturates
     and recedes. I translate this into `.spec-panel.is-winner` (lit, accent-glow edge,
     raised) vs `.is-loser` (desaturated, dimmed, dropped back a plane).
   - Borrowed: a heavy **CJK result stamp** (勝 "win") rendered large and angled. I reuse
     the existing `.win-mark` 勝 but scale it into a proper plaque on the winning panel.

2. **Tekken 8 — round/result UI**
   https://www.tekken.com/en-us
   - Borrowed: **layered depth on the side cards** — a dark base plate, a slightly lighter
     inner surface, and a single colored energy edge per fighter (not a full colored card).
     My panels keep the existing 4px team edge but add an inner top-light gradient + inset
     hairline so the card reads as a physical lacquered plate, not a flat rectangle.

3. **OP.GG / League post-game scoreboard**
   https://op.gg/
   - Borrowed: **win = a tinted full-row treatment, loss = neutral**. Rather than coloring
     everything, only the outcome state changes the surface tint. I apply a faint team-tint
     wash to the winning panel background and leave the loser on the neutral plate.
   - Borrowed: **tabular-num stat rows with a quiet key/value rhythm** — already present in
     the viewer's `.spec-row`; I tighten the divider contrast and key/value weight pairing.

4. **YouTube / Vimeo / mpv (osc) media scrubbers**
   https://www.youtube.com/ · https://vimeo.com/
   - Borrowed: a **real played-vs-remaining track**. Stock `<input type=range>` shows one
     flat bar; premium players show filled progress in the accent color behind the thumb.
     I build this with a CSS variable `--scrub-pct` driving a `linear-gradient` track (warm
     amber filled / dark unfilled) plus a glowing thumb that grows on hover/active.
   - Borrowed: the **thumb only appears alive on hover** — calm at rest, larger + ringed on
     interaction. Keeps the resting UI quiet (premium restraint) but responsive.

5. **Riot / VALORANT broadcast lower-thirds + agent select**
   https://playvalorant.com/
   - Borrowed: **display type does the heavy lifting**. A condensed/serif display face for
     the marquee word, tight tracking, and a clear type scale step between marquee, title,
     and supporting text. I bring in a tasteful Google display face for the brand row and
     enforce a deliberate scale (50 → 28 → 21 → 14 → 11px) instead of muddy in-between sizes.
   - Borrowed: **accent is rationed**. Orange/gold appears on exactly the things that matter
     (kanji, the live Play control, the scrubber fill, the winner glow). Everything else is
     ink + washi neutrals, so the accent reads as energy, not decoration.

---

## What makes these feel premium (and how the theme applies it)

| Premium signal | Reference | Move in `theme.css` |
|---|---|---|
| **Type scale + display face** | VALORANT, SF6 | Google Fonts display serif on `.kanji`; condensed sans for `header h1`; locked 50/28/21/14/11 scale; tabular nums on stats. |
| **Glass / layered depth** | Tekken 8, Tekken | Panels & stage get a base plate + inner top-light gradient + inset hairline + outer drop shadow → reads as a real surface, not a div. |
| **Rationed accent** | VALORANT | Orange/gold only on kanji, Play, scrubber fill, winner glow. Neutrals everywhere else. |
| **Winner emphasis** | SF6, OP.GG | `.is-winner` lifts, warms, glows, shows a 勝 plaque; `.is-loser` desaturates + recedes. |
| **Real scrubber** | YouTube/Vimeo | Played/remaining split via `--scrub-pct` gradient; thumb grows + glows on hover/active. |
| **Motion that's structural** | SF6 result | Panels animate to their winner/loser state; controls have crisp hover lift + active press; **no blanket fade-up**. Honors `prefers-reduced-motion`. |

---

## Drop-in instructions

`theme.css` is written to layer **on top of** the existing `<style>` block in
`viewer.html` (it intentionally re-declares the same selectors with stronger values, and
uses a couple of low-specificity overrides). To apply it to the real viewer later:

```html
<!-- in viewer.html <head>, AFTER the existing <style>...</style> block -->
<link rel="stylesheet" href="theme.css" />
```

It works with **zero markup changes** — every existing selector
(`header`, `.kanji`, `.spec-panel`, `.spec-row`, `.side-pill`, `.transport-cluster .tb`,
`.transport input[type=range]`, etc.) is styled as-is.

### Optional class hooks for the *full* effect

Two enhancements need a tiny, additive JS hook (each is one line, no behavior change).
The theme degrades gracefully if you skip them.

1. **Winner / loser panel emphasis** — add `is-winner` / `is-loser` to the panel element.
   In the existing `renderSpec(side, ...)` where `oc` is already computed:
   ```js
   const panel = document.getElementById("panel" + side); // panelL / panelR
   panel.classList.toggle("is-winner", oc === "win");
   panel.classList.toggle("is-loser",  oc === "loss");
   // draw => neither class (both panels stay neutral)
   ```
   Without this, panels still get the upgraded plate/depth look; they just won't do the
   win/loss lift-and-dim.

2. **Filled scrubber track** — set a CSS var as the scrub value changes so the played
   portion fills. In the existing `scrub` input handler / wherever `scrub.value` is set:
   ```js
   function paintScrub() {
     const pct = scrub.max > 0 ? (scrub.value / scrub.max) * 100 : 0;
     scrub.style.setProperty("--scrub-pct", pct + "%");
   }
   // call paintScrub() on 'input' and after programmatic seeks
   ```
   Without this, the scrubber still gets the styled track + glowing thumb; the fill just
   won't track playback (defaults to 0%).

### Class hooks the theme expects (already in the markup — no change needed)

- `header`, `.brand`, `.kanji`, `header h1`, `header .sub`, `.legend`
- `.controls-bar`, `.controls-bar .lbl`, `select`, `.btn-import`, `.transport-cluster .tb`,
  `.transport-cluster .tb.play`
- `.arena`, `.spec-panel.left` / `.right`, `.panel-head`, `.role`, `.name`, `.sensei`,
  `.avatar`, `.side-pill.trained` / `.base`, `.spec-rows`, `.spec-row .k/.v`, `.win-mark`,
  `.details-btn`
- `.stage-wrap`, `.canvas-frame`, `#canvas`, `.transport`, `.ts`, `input[type=range]#scrub`,
  `.fs-btn`

### Tokens added by the theme (safe to reuse / override)

The theme adds a small token layer on top of the existing `:root` palette: refined
neutrals, an `--accent-orange` / `--accent-gold`, team-tint rgb channels for glows
(`--trained-rgb`, `--base-rgb`), elevation shadows (`--elev-1/2/3`), and the scrubber's
`--scrub-pct`. It never *removes* an existing token, so the viewer JS that reads
`--trained` / `--base` / `--accent` via `getCSS()` keeps working unchanged.

---

## Verification

- `theme_preview.html` is self-contained (links `theme.css`, uses static placeholder
  data, no fetch, no build step). Open it directly via `file://` to review the upgrade.
- The preview shows: header + brand + legend, the full controls/transport bar, both spec
  panels (one rendered as **winner**, one as **loser** to show the emphasis treatment),
  the stage frame, and a styled scrubber at a mid-playback position.
