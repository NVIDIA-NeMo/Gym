/* @ds-bundle: {"format":4,"namespace":"BroadsheetDesignSystem_b6f617","components":[{"name":"Button","sourcePath":"components/actions/Button.jsx"},{"name":"Tag","sourcePath":"components/actions/Tag.jsx"},{"name":"Table","sourcePath":"components/data/Table.jsx"},{"name":"Field","sourcePath":"components/forms/Field.jsx"},{"name":"Input","sourcePath":"components/forms/Input.jsx"},{"name":"Radio","sourcePath":"components/forms/Radio.jsx"},{"name":"RadioGroup","sourcePath":"components/forms/Radio.jsx"},{"name":"Segmented","sourcePath":"components/forms/Segmented.jsx"},{"name":"PHOSPHOR_DUOTONE","sourcePath":"components/icons/Icon.jsx"},{"name":"Icon","sourcePath":"components/icons/Icon.jsx"},{"name":"Nav","sourcePath":"components/navigation/Nav.jsx"},{"name":"Cmyk","sourcePath":"components/print/Cmyk.jsx"},{"name":"Halftone","sourcePath":"components/print/Halftone.jsx"},{"name":"HeadRule","sourcePath":"components/print/HeadRule.jsx"},{"name":"Dateline","sourcePath":"components/print/HeadRule.jsx"},{"name":"PlateText","sourcePath":"components/print/PlateText.jsx"},{"name":"Card","sourcePath":"components/surfaces/Card.jsx"},{"name":"Dialog","sourcePath":"components/surfaces/Dialog.jsx"}],"sourceHashes":{"assets/print-plates.js":"00209245728b","components/actions/Button.jsx":"fc1292e991da","components/actions/Tag.jsx":"89c53545e422","components/data/Table.jsx":"e276129b65aa","components/forms/Field.jsx":"f63c5f95cf58","components/forms/Input.jsx":"f2acf42381a8","components/forms/Radio.jsx":"0aeb39071c59","components/forms/Segmented.jsx":"202225c3c886","components/icons/Icon.jsx":"08538cc0149b","components/navigation/Nav.jsx":"426cad035673","components/print/Cmyk.jsx":"f23029da45b7","components/print/Halftone.jsx":"db33dd45cc28","components/print/HeadRule.jsx":"03adb8aa58eb","components/print/PlateText.jsx":"b876fe30078b","components/surfaces/Card.jsx":"e555a0f159ed","components/surfaces/Dialog.jsx":"31b41a71df0e","ui_kits/daybreak/Issues.jsx":"168182e0387c","ui_kits/daybreak/Landing.jsx":"917224affc32","ui_kits/daybreak/Preferences.jsx":"2a4d6741d703"},"inlinedExternals":[],"unexposedExports":[]} */

(() => {

const __ds_ns = (window.BroadsheetDesignSystem_b6f617 = window.BroadsheetDesignSystem_b6f617 || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// assets/print-plates.js
try { (() => {
// print-plates.js — the Broadsheet separation filters, shipped with the
// system. Each single-plate filter extracts one process plate from a
// photograph, rendered as that ink on the sheet — cyan from R, magenta
// from G, yellow from B, and a 60%-strength luminance K printed in the
// text ink (#201e1d). Values are the system's own inks (accent, accent-2,
// process-yellow, text). #sep-all chains the four into ONE compound
// filter for a single swappable image (see .cmyk .print in styles.css):
// each stage re-extracts a plate from SourceGraphic, clips it to the
// source's own silhouette (feComposite operator="in" against
// SourceAlpha), and offsets the clipped sheet by the registered
// misregistration (C 0,0 / M 5,3 / Y -5,-3 / K 3,6); the sheets multiply
// where they cross and show alone where they don't. The clip is why
// there is no paper flood here anymore (divergence round 3, Barron: "it
// crops the effect"): an unclipped feColorMatrix lays its constant-term
// ink across the whole filter region, which forced overflow:hidden on
// the figure and guillotined the misregistration flat at the box edge.
// Clipped sheets paint nothing outside themselves, so the figure can let
// them overhang and the offsets read at the edges, the films askew on
// the stack. Each sheet carries its own white (the matrices print ink
// on the SOURCE's whites now, not the paper token), so the multiply
// needs no backdrop — and dropping the flood's #f3f2f2 multiply
// brightens every #sep-all composite by that factor (~5%, measured
// before/after on the landing split's interior) as part of the same
// goal the ink purification below finishes: the resting print sits
// closer to the photograph, and the hover resolve lands on it exactly.
//
// The offsets live on feOffset primitives tagged data-plate, because the
// press driver below animates them: hover gathers the plates into
// register (the films squared on the light table — the deck's
// pulled-apart slide established the move, from the Catalogue Three
// reference) while the same eased value purifies the plate inks — each
// feColorMatrix (tagged data-plate-mat) lerps from its brand separation
// to the pure-process factorization whose four plates multiply back to
// SourceGraphic exactly — so the converged MERGE is the photograph: the
// brand inks and the K plate are what make the resting print denser
// than the source, which is why the earlier films-lift end-swap read
// as a brightness pop (divergence round 4, Barron: the merged plates
// "should look normal"). The screen never shows anything but the
// four-plate multiply; nothing is swapped in at any point.
// The pointer leans the moving plates a breath toward the cursor as it
// roams the page (divergence round 3, Barron: "slightly adjust the
// offsets based on your mouse position"). The driver also
// publishes the pointer as bare -1..1 factors (--press-nx/--press-ny)
// on the root, which the text-plate treatments (.cmyk-num, .cmyk-head
// in styles.css) multiply into their own em spreads. One set of defs serves the
// document, so a hover on any one .print figure drives them all — pages
// show one compound-filter figure at a time (the landing's split, one
// deck slide), which keeps that a non-event (the deck edit rail's live
// thumbnails resolve the same defs and move in sympathy — previews,
// previewing). The driver stands down
// wholesale under prefers-reduced-motion or without a fine hover
// pointer; styles.css carries the matching media-gated :hover cut as
// the fallback, so the reduced-motion experience is exactly the old
// one.
//
// This file exists because the defs must be IN the document: a data-URI
// filter reference does not survive Chromium (probed, review round 3),
// and external-file references are unreliable across engines — script
// injection puts the defs in the document itself, the one mechanism the
// inline block already proved. The compiled bundle (_ds_bundle.js)
// inlines this file, so every template page gets the defs through
// ds-base.js's bundle load (ds-base.js itself stays identical across
// the six systems — a repo-wide invariant); a page that links
// styles.css directly instead adds
// <script src="print-plates.js"></script> (path relative to the
// system root, as for styles.css). Injected at the end of <body>, outside
// any component tree, so no single section's deletion can strand the
// references — filter references resolve document-wide.
(() => {
  const ID = 'broadsheet-print-plates';
  const mount = () => {
    if (document.getElementById(ID)) return; // idempotent — load twice, inject once
    const host = document.createElement('div');
    host.innerHTML = `<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
  <filter id="sep-c" color-interpolation-filters="sRGB"><feColorMatrix type="matrix" values="1 0 0 0 0  0.467 0 0 0 0.533  0.310 0 0 0 0.690  0 0 0 0 1"/></filter>
  <filter id="sep-m" color-interpolation-filters="sRGB"><feColorMatrix type="matrix" values="0 0.161 0 0 0.839  0 1 0 0 0  0 0.576 0 0 0.424  0 0 0 0 1"/></filter>
  <filter id="sep-y" color-interpolation-filters="sRGB"><feColorMatrix type="matrix" values="0 0 0.071 0 0.929  0 0 0.267 0 0.733  0 0 1 0 0  0 0 0 0 1"/></filter>
  <filter id="sep-k" color-interpolation-filters="sRGB"><feColorMatrix type="matrix" values="0.112 0.375 0.038 0 0.475  0.113 0.379 0.038 0 0.471  0.113 0.380 0.038 0 0.468  0 0 0 0 1"/></filter>
  <filter id="sep-all" color-interpolation-filters="sRGB">
    <feColorMatrix in="SourceGraphic" type="matrix" values="1 0 0 0 0  0.467 0 0 0 0.533  0.310 0 0 0 0.690  0 0 0 0 1" data-plate-mat="c" result="c0"/>
    <feComposite in="c0" in2="SourceAlpha" operator="in" result="c"/>
    <feColorMatrix in="SourceGraphic" type="matrix" values="0 0.161 0 0 0.839  0 1 0 0 0  0 0.576 0 0 0.424  0 0 0 0 1" data-plate-mat="m" result="m0"/>
    <feComposite in="m0" in2="SourceAlpha" operator="in" result="m1"/>
    <feOffset in="m1" dx="5" dy="3" data-plate="m" result="m"/>
    <feColorMatrix in="SourceGraphic" type="matrix" values="0 0 0.071 0 0.929  0 0 0.267 0 0.733  0 0 1 0 0  0 0 0 0 1" data-plate-mat="y" result="y0"/>
    <feComposite in="y0" in2="SourceAlpha" operator="in" result="y1"/>
    <feOffset in="y1" dx="-5" dy="-3" data-plate="y" result="y"/>
    <feColorMatrix in="SourceGraphic" type="matrix" values="0.112 0.375 0.038 0 0.475  0.113 0.379 0.038 0 0.471  0.113 0.380 0.038 0 0.468  0 0 0 0 1" data-plate-mat="k" result="k0"/>
    <feComposite in="k0" in2="SourceAlpha" operator="in" result="k1"/>
    <feOffset in="k1" dx="3" dy="6" data-plate="k" result="k"/>
    <feBlend in="m" in2="c" mode="multiply" result="s1"/>
    <feBlend in="y" in2="s1" mode="multiply" result="s2"/>
    <feBlend in="k" in2="s2" mode="multiply"/>
  </filter>
</defs></svg>`;
    const svg = host.firstChild;
    svg.id = ID;
    document.body.appendChild(svg);
    press(svg);
  };

  // The press driver — hover registration and the pointer lean. All
  // numbers are the measured design values, not tunables-in-waiting:
  // LEAN is ±2.5px x / ±2px y at the viewport edges (half the M/Y
  // x-offset, two-thirds of its y — a breath, per "slightly"); the
  // text plates take the
  // same gesture at glyph scale, so the driver publishes the bare
  // pointer factors (--press-nx/--press-ny, -1..1) and each text
  // construction multiplies them into its own em spread in styles.css;
  // REGISTER_MS matches the deck's gather without its theatre (the
  // 1.4s slide move reads as a scene, a hover should answer).
  const press = svg => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (!matchMedia('(hover: hover) and (pointer: fine)').matches) return;
    const BASE = {
      m: [5, 3],
      y: [-5, -3],
      k: [3, 6]
    };
    const LEAN_PX = [2.5, 2],
      REGISTER_MS = 450;
    const nodes = {};
    svg.querySelectorAll('feOffset[data-plate]').forEach(n => {
      nodes[n.dataset.plate] = n;
    });
    // The plate matrices' two endpoints. INK is the brand separation the
    // sheet prints at rest (the values in the defs above). TRUE is the
    // pure-process factorization — C passes R and floods G,B to 1, M and
    // Y likewise for their channels, K goes to white (the multiply
    // identity) — chosen because the four TRUE plates multiply back to
    // SourceGraphic EXACTLY: (R,1,1)·(1,G,1)·(1,1,B)·(1,1,1) = (R,G,B).
    // The brand inks and the K plate are exactly what makes the resting
    // print denser than the photograph, so easing each matrix INK→TRUE
    // as the plates converge lands the merged print ON the photograph —
    // the image on screen is a four-plate multiply at every instant, and
    // the converged merge "looks normal" by algebra, not by swapping
    // anything in at the end (divergence round 4, Barron: the old
    // films-lift end-swap read as a brightness pop).
    const mats = {};
    svg.querySelectorAll('feColorMatrix[data-plate-mat]').forEach(n => {
      mats[n.dataset.plateMat] = n;
    });
    const INK = {
      c: [1, 0, 0, 0, 0, 0.467, 0, 0, 0, 0.533, 0.310, 0, 0, 0, 0.690, 0, 0, 0, 0, 1],
      m: [0, 0.161, 0, 0, 0.839, 0, 1, 0, 0, 0, 0, 0.576, 0, 0, 0.424, 0, 0, 0, 0, 1],
      y: [0, 0, 0.071, 0, 0.929, 0, 0, 0.267, 0, 0.733, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
      k: [0.112, 0.375, 0.038, 0, 0.475, 0.113, 0.379, 0.038, 0, 0.471, 0.113, 0.380, 0.038, 0, 0.468, 0, 0, 0, 0, 1]
    };
    const TRUE_ = {
      c: [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      m: [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      y: [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
      k: [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    };
    const root = document.documentElement;
    let nx = 0,
      ny = 0; // smoothed pointer, -1..1 from viewport center
    let tx = 0,
      ty = 0; // raw pointer target
    let reg = 1; // 1 = misregistered (rest), 0 = in register
    let regFrom = 1,
      regTo = 1,
      regT0 = 0;
    let raf = 0,
      lastOffs = '',
      lastReg = -1,
      lastProps = '';
    const ease = t => 1 - Math.pow(1 - t, 3); // cubic out
    const tick = now => {
      raf = 0;
      nx += (tx - nx) * 0.22;
      ny += (ty - ny) * 0.22; // soften the hand
      if (regTo !== reg || regT0) {
        const t = Math.min(1, (now - regT0) / REGISTER_MS);
        reg = regFrom + (regTo - regFrom) * ease(t);
        if (t >= 1) {
          reg = regTo;
          regT0 = 0;
        }
      }
      // every write below is guarded on its COMPUTED output, not its
      // inputs — an equal-value setAttribute still dirties the filter,
      // and at full register the offsets are 0.00 whatever the lean, so
      // a pointer roaming over a converged figure must not recompute
      // the compound filter every frame
      const lx = LEAN_PX[0] * nx,
        ly = LEAN_PX[1] * ny;
      const vals = {};
      let offsKey = '';
      for (const p in BASE) {
        const dx = ((BASE[p][0] + lx) * reg).toFixed(2);
        const dy = ((BASE[p][1] + ly) * reg).toFixed(2);
        vals[p] = [dx, dy];
        offsKey += dx + ',' + dy + ';';
      }
      if (offsKey !== lastOffs) {
        lastOffs = offsKey;
        for (const p in BASE) {
          nodes[p].setAttribute('dx', vals[p][0]);
          nodes[p].setAttribute('dy', vals[p][1]);
        }
      }
      // the ink purification rides the same eased value — INK at rest
      // (reg 1), TRUE at register (reg 0); see the endpoint tables above
      if (reg !== lastReg) {
        lastReg = reg;
        for (const p in mats) {
          const a = INK[p],
            b = TRUE_[p],
            v = new Array(20);
          for (let i = 0; i < 20; i++) v[i] = (b[i] + (a[i] - b[i]) * reg).toFixed(3);
          mats[p].setAttribute('values', v.join(' '));
        }
      }
      // the text plates lean whatever the photo's register state — the
      // props move with the pointer alone
      const pk = nx.toFixed(3) + ',' + ny.toFixed(3);
      if (pk !== lastProps) {
        lastProps = pk;
        root.style.setProperty('--press-nx', nx.toFixed(3));
        root.style.setProperty('--press-ny', ny.toFixed(3));
      }
      if (regT0 || Math.abs(tx - nx) > 0.002 || Math.abs(ty - ny) > 0.002) schedule();
    };
    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(tick);
    };
    addEventListener('pointermove', e => {
      tx = 2 * e.clientX / innerWidth - 1;
      ty = 2 * e.clientY / innerHeight - 1;
      schedule();
    }, {
      passive: true
    });
    const retarget = to => {
      regFrom = reg;
      regTo = to;
      regT0 = performance.now();
      schedule();
    };
    document.addEventListener('pointerover', e => {
      const p = e.target.closest && e.target.closest('.cmyk .print');
      if (p && !(e.relatedTarget && p.contains(e.relatedTarget))) retarget(0);
    });
    document.addEventListener('pointerout', e => {
      const p = e.target.closest && e.target.closest('.cmyk .print');
      if (p && !(e.relatedTarget && p.contains(e.relatedTarget))) retarget(1);
    });
  };
  if (document.body) mount();else addEventListener('DOMContentLoaded', mount);
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "assets/print-plates.js", error: String((e && e.message) || e) }); }

// components/actions/Button.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
const cx = (...a) => a.filter(Boolean).join(' ');
function Button({
  variant = 'primary',
  icon = false,
  block = false,
  as = 'button',
  className,
  children,
  ...rest
}) {
  const Tag = as;
  const cls = cx('btn', `btn-${variant}`, icon && 'btn-icon', block && 'btn-block', className);
  return /*#__PURE__*/React.createElement(Tag, _extends({
    type: Tag === 'button' ? 'button' : undefined,
    className: cls
  }, rest), children);
}
Object.assign(__ds_scope, { Button });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/actions/Button.jsx", error: String((e && e.message) || e) }); }

// components/actions/Tag.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Tag({
  tone = 'accent',
  className,
  children,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("span", _extends({
    className: ['tag', `tag-${tone}`, className].filter(Boolean).join(' ')
  }, rest), children);
}
Object.assign(__ds_scope, { Tag });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/actions/Tag.jsx", error: String((e && e.message) || e) }); }

// components/data/Table.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Table({
  columns,
  rows,
  className,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("table", _extends({
    className: ['table', className].filter(Boolean).join(' ')
  }, rest), /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, columns.map(c => /*#__PURE__*/React.createElement("th", {
    key: c.key,
    style: c.align ? {
      textAlign: c.align
    } : undefined
  }, c.label)))), /*#__PURE__*/React.createElement("tbody", null, rows.map((r, i) => /*#__PURE__*/React.createElement("tr", {
    key: r.id ?? i
  }, columns.map(c => /*#__PURE__*/React.createElement("td", {
    key: c.key,
    className: c.muted ? 'text-muted' : undefined,
    style: c.align ? {
      textAlign: c.align
    } : undefined
  }, c.render ? c.render(r) : r[c.key]))))));
}
Object.assign(__ds_scope, { Table });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/data/Table.jsx", error: String((e && e.message) || e) }); }

// components/forms/Field.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Field({
  label,
  htmlFor,
  labelId,
  className,
  children,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("div", _extends({
    className: ['field', className].filter(Boolean).join(' ')
  }, rest), label && /*#__PURE__*/React.createElement("label", {
    htmlFor: htmlFor,
    id: labelId
  }, label), children);
}
Object.assign(__ds_scope, { Field });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Field.jsx", error: String((e && e.message) || e) }); }

// components/forms/Input.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Input({
  multiline = false,
  className,
  ...rest
}) {
  const cls = ['input', className].filter(Boolean).join(' ');
  return multiline ? /*#__PURE__*/React.createElement("textarea", _extends({
    className: cls
  }, rest)) : /*#__PURE__*/React.createElement("input", _extends({
    className: cls
  }, rest));
}
Object.assign(__ds_scope, { Input });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Input.jsx", error: String((e && e.message) || e) }); }

// components/forms/Radio.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Radio({
  children,
  className,
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("label", {
    className: ['radio', className].filter(Boolean).join(' '),
    style: style
  }, /*#__PURE__*/React.createElement("input", _extends({
    type: "radio"
  }, rest)), /*#__PURE__*/React.createElement("span", {
    className: "dot"
  }), children);
}
function RadioGroup({
  label,
  name,
  options,
  value,
  defaultValue,
  onChange,
  id
}) {
  const [v, setV] = React.useState(defaultValue ?? options[0]?.value);
  const cur = value ?? v;
  const lid = id || `rg-${name}`;
  return /*#__PURE__*/React.createElement("div", {
    className: "field"
  }, label && /*#__PURE__*/React.createElement("label", {
    id: lid
  }, label), /*#__PURE__*/React.createElement("div", {
    role: "radiogroup",
    "aria-labelledby": label ? lid : undefined,
    style: {
      display: 'grid',
      gap: 'var(--space-1)'
    }
  }, options.map(o => /*#__PURE__*/React.createElement(Radio, {
    key: o.value,
    name: name,
    value: o.value,
    checked: cur === o.value,
    onChange: () => {
      setV(o.value);
      onChange && onChange(o.value);
    }
  }, o.label))));
}
Object.assign(__ds_scope, { Radio, RadioGroup });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Radio.jsx", error: String((e && e.message) || e) }); }

// components/forms/Segmented.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Segmented({
  name,
  options,
  value,
  defaultValue,
  onChange,
  className,
  ...rest
}) {
  const [v, setV] = React.useState(defaultValue ?? options[0]?.value);
  const cur = value ?? v;
  return /*#__PURE__*/React.createElement("div", _extends({
    className: ['seg', className].filter(Boolean).join(' '),
    role: "radiogroup"
  }, rest), options.map(o => /*#__PURE__*/React.createElement("label", {
    className: "seg-opt",
    key: o.value
  }, /*#__PURE__*/React.createElement("input", {
    type: "radio",
    name: name,
    value: o.value,
    checked: cur === o.value,
    onChange: () => {
      setV(o.value);
      onChange && onChange(o.value);
    }
  }), o.icon, o.label)));
}
Object.assign(__ds_scope, { Segmented });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Segmented.jsx", error: String((e && e.message) || e) }); }

// components/icons/Icon.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
// Phosphor duotone glyphs (viewBox 0 0 256 256), copied from the Broadsheet source. Duotone = a 25%-opacity fill plate under the solid stroke plate; both on currentColor.
const PHOSPHOR_DUOTONE = {
  sparkle: ['M194.82,151.43l-55.09,20.3-20.3,55.09a7.92,7.92,0,0,1-14.86,0l-20.3-55.09-55.09-20.3a7.92,7.92,0,0,1,0-14.86l55.09-20.3,20.3-55.09a7.92,7.92,0,0,1,14.86,0l20.3,55.09,55.09,20.3A7.92,7.92,0,0,1,194.82,151.43Z', 'M197.58,129.06,146,110l-19-51.62a15.92,15.92,0,0,0-29.88,0L78,110l-51.62,19a15.92,15.92,0,0,0,0,29.88L78,178l19,51.62a15.92,15.92,0,0,0,29.88,0L146,178l51.62-19a15.92,15.92,0,0,0,0-29.88ZM137,164.22a8,8,0,0,0-4.74,4.74L112,223.85,91.78,169A8,8,0,0,0,87,164.22L32.15,144,87,123.78A8,8,0,0,0,91.78,119L112,64.15,132.22,119a8,8,0,0,0,4.74,4.74L191.85,144ZM144,40a8,8,0,0,1,8-8h16V16a8,8,0,0,1,16,0V32h16a8,8,0,0,1,0,16H184V64a8,8,0,0,1-16,0V48H152A8,8,0,0,1,144,40ZM248,88a8,8,0,0,1-8,8h-8v8a8,8,0,0,1-16,0V96h-8a8,8,0,0,1,0-16h8V72a8,8,0,0,1,16,0v8h8A8,8,0,0,1,248,88Z'],
  layers: ['M224,80l-96,56L32,80l96-56Z', 'M230.91,172A8,8,0,0,1,228,182.91l-96,56a8,8,0,0,1-8.06,0l-96-56A8,8,0,0,1,36,169.09l92,53.65,92-53.65A8,8,0,0,1,230.91,172ZM220,121.09l-92,53.65L36,121.09A8,8,0,0,0,28,134.91l96,56a8,8,0,0,0,8.06,0l96-56A8,8,0,1,0,220,121.09ZM24,80a8,8,0,0,1,4-6.91l96-56a8,8,0,0,1,8.06,0l96,56a8,8,0,0,1,0,13.82l-96,56a8,8,0,0,1-8.06,0l-96-56A8,8,0,0,1,24,80Zm23.88,0L128,126.74,208.12,80,128,33.26Z'],
  circle: ['M224,128a96,96,0,1,1-96-96A96,96,0,0,1,224,128Z', 'M128,24A104,104,0,1,0,232,128,104.11,104.11,0,0,0,128,24Zm0,192a88,88,0,1,1,88-88A88.1,88.1,0,0,1,128,216Z'],
  arrow: ['M216,128l-72,72V56Z', 'M221.66,122.34l-72-72A8,8,0,0,0,136,56v64H40a8,8,0,0,0,0,16h96v64a8,8,0,0,0,13.66,5.66l72-72A8,8,0,0,0,221.66,122.34ZM152,180.69V75.31L204.69,128Z'],
  search: ['M192,112a80,80,0,1,1-80-80A80,80,0,0,1,192,112Z', 'M229.66,218.34,179.6,168.28a88.21,88.21,0,1,0-11.32,11.31l50.06,50.07a8,8,0,0,0,11.32-11.32ZM40,112a72,72,0,1,1,72,72A72.08,72.08,0,0,1,40,112Z'],
  settings: ['M207.86,123.18l16.78-21a99.14,99.14,0,0,0-10.07-24.29l-26.7-3a81,81,0,0,0-6.81-6.81l-3-26.71a99.43,99.43,0,0,0-24.3-10l-21,16.77a81.59,81.59,0,0,0-9.64,0l-21-16.78A99.14,99.14,0,0,0,77.91,41.43l-3,26.7a81,81,0,0,0-6.81,6.81l-26.71,3a99.43,99.43,0,0,0-10,24.3l16.77,21a81.59,81.59,0,0,0,0,9.64l-16.78,21a99.14,99.14,0,0,0,10.07,24.29l26.7,3a81,81,0,0,0,6.81,6.81l3,26.71a99.43,99.43,0,0,0,24.3,10l21-16.77a81.59,81.59,0,0,0,9.64,0l21,16.78a99.14,99.14,0,0,0,24.29-10.07l3-26.7a81,81,0,0,0,6.81-6.81l26.71-3a99.43,99.43,0,0,0,10-24.3l-16.77-21A81.59,81.59,0,0,0,207.86,123.18ZM128,168a40,40,0,1,1,40-40A40,40,0,0,1,128,168Z', 'M128,80a48,48,0,1,0,48,48A48.05,48.05,0,0,0,128,80Zm0,80a32,32,0,1,1,32-32A32,32,0,0,1,128,160Zm88-29.84q.06-2.16,0-4.32l14.92-18.64a8,8,0,0,0,1.48-7.06,107.6,107.6,0,0,0-10.88-26.25,8,8,0,0,0-6-3.93l-23.72-2.64q-1.48-1.56-3-3L186,40.54a8,8,0,0,0-3.94-6,107.29,107.29,0,0,0-26.25-10.86,8,8,0,0,0-7.06,1.48L130.16,40Q128,40,125.84,40L107.2,25.11a8,8,0,0,0-7.06-1.48A107.6,107.6,0,0,0,73.89,34.51a8,8,0,0,0-3.93,6L67.32,64.27q-1.56,1.49-3,3L40.54,70a8,8,0,0,0-6,3.94,107.71,107.71,0,0,0-10.87,26.25,8,8,0,0,0,1.49,7.06L40,125.84Q40,128,40,130.16L25.11,148.8a8,8,0,0,0-1.48,7.06,107.6,107.6,0,0,0,10.88,26.25,8,8,0,0,0,6,3.93l23.72,2.64q1.49,1.56,3,3L70,215.46a8,8,0,0,0,3.94,6,107.71,107.71,0,0,0,26.25,10.87,8,8,0,0,0,7.06-1.49L125.84,216q2.16.06,4.32,0l18.64,14.92a8,8,0,0,0,7.06,1.48,107.21,107.21,0,0,0,26.25-10.88,8,8,0,0,0,3.93-6l2.64-23.72q1.56-1.48,3-3L215.46,186a8,8,0,0,0,6-3.94,107.71,107.71,0,0,0,10.87-26.25,8,8,0,0,0-1.49-7.06Zm-16.1-6.5a73.93,73.93,0,0,1,0,8.68,8,8,0,0,0,1.74,5.48l14.19,17.73a91.57,91.57,0,0,1-6.23,15L187,173.11a8,8,0,0,0-5.1,2.64,74.11,74.11,0,0,1-6.14,6.14,8,8,0,0,0-2.64,5.1l-2.51,22.58a91.32,91.32,0,0,1-15,6.23l-17.74-14.19a8,8,0,0,0-5-1.75h-.48a73.93,73.93,0,0,1-8.68,0,8.06,8.06,0,0,0-5.48,1.74L100.45,215.8a91.57,91.57,0,0,1-15-6.23L82.89,187a8,8,0,0,0-2.64-5.1,74.11,74.11,0,0,1-6.14-6.14,8,8,0,0,0-5.1-2.64L46.43,170.6a91.32,91.32,0,0,1-6.23-15l14.19-17.74a8,8,0,0,0,1.74-5.48,73.93,73.93,0,0,1,0-8.68,8,8,0,0,0-1.74-5.48L40.2,100.45a91.57,91.57,0,0,1,6.23-15L69,82.89a8,8,0,0,0,5.1-2.64,74.11,74.11,0,0,1,6.14-6.14A8,8,0,0,0,82.89,69L85.4,46.43a91.32,91.32,0,0,1,15-6.23l17.74,14.19a8,8,0,0,0,5.48,1.74,73.93,73.93,0,0,1,8.68,0,8.06,8.06,0,0,0,5.48-1.74L155.55,40.2a91.57,91.57,0,0,1,15,6.23L173.11,69a8,8,0,0,0,2.64,5.1,74.11,74.11,0,0,1,6.14,6.14,8,8,0,0,0,5.1,2.64l22.58,2.51a91.32,91.32,0,0,1,6.23,15l-14.19,17.74A8,8,0,0,0,199.87,123.66Z'],
  user: ['M192,96a64,64,0,1,1-64-64A64,64,0,0,1,192,96Z', 'M230.92,212c-15.23-26.33-38.7-45.21-66.09-54.16a72,72,0,1,0-73.66,0C63.78,166.78,40.31,185.66,25.08,212a8,8,0,1,0,13.85,8c18.84-32.56,52.14-52,89.07-52s70.23,19.44,89.07,52a8,8,0,1,0,13.85-8ZM72,96a56,56,0,1,1,56,56A56.06,56.06,0,0,1,72,96Z'],
  heart: ['M232,102c0,66-104,122-104,122S24,168,24,102A54,54,0,0,1,78,48c22.59,0,41.94,12.31,50,32,8.06-19.69,27.41-32,50-32A54,54,0,0,1,232,102Z', 'M178,40c-20.65,0-38.73,8.88-50,23.89C116.73,48.88,98.65,40,78,40a62.07,62.07,0,0,0-62,62c0,70,103.79,126.66,108.21,129a8,8,0,0,0,7.58,0C136.21,228.66,240,172,240,102A62.07,62.07,0,0,0,178,40ZM128,214.8C109.74,204.16,32,155.69,32,102A46.06,46.06,0,0,1,78,56c19.45,0,35.78,10.36,42.6,27a8,8,0,0,0,14.8,0c6.82-16.67,23.15-27,42.6-27a46.06,46.06,0,0,1,46,46C224,155.61,146.24,204.15,128,214.8Z'],
  bell: ['M208,192H48a8,8,0,0,1-6.88-12C47.71,168.6,56,139.81,56,104a72,72,0,0,1,144,0c0,35.82,8.3,64.6,14.9,76A8,8,0,0,1,208,192Z', 'M221.8,175.94C216.25,166.38,208,139.33,208,104a80,80,0,1,0-160,0c0,35.34-8.26,62.38-13.81,71.94A16,16,0,0,0,48,200H88.81a40,40,0,0,0,78.38,0H208a16,16,0,0,0,13.8-24.06ZM128,216a24,24,0,0,1-22.62-16h45.24A24,24,0,0,1,128,216ZM48,184c7.7-13.24,16-43.92,16-80a64,64,0,1,1,128,0c0,36.05,8.28,66.73,16,80Z'],
  calendar: ['M216,48V88H40V48a8,8,0,0,1,8-8H208A8,8,0,0,1,216,48Z', 'M208,32H184V24a8,8,0,0,0-16,0v8H88V24a8,8,0,0,0-16,0v8H48A16,16,0,0,0,32,48V208a16,16,0,0,0,16,16H208a16,16,0,0,0,16-16V48A16,16,0,0,0,208,32ZM72,48v8a8,8,0,0,0,16,0V48h80v8a8,8,0,0,0,16,0V48h24V80H48V48ZM208,208H48V96H208V208Zm-96-88v64a8,8,0,0,1-16,0V132.94l-4.42,2.22a8,8,0,0,1-7.16-14.32l16-8A8,8,0,0,1,112,120Zm59.16,30.45L152,176h16a8,8,0,0,1,0,16H136a8,8,0,0,1-6.4-12.8l28.78-38.37A8,8,0,1,0,145.07,132a8,8,0,1,1-13.85-8A24,24,0,0,1,176,136,23.76,23.76,0,0,1,171.16,150.45Z'],
  image: ['M224,56V178.06l-39.72-39.72a8,8,0,0,0-11.31,0L147.31,164,97.66,114.34a8,8,0,0,0-11.32,0L32,168.69V56a8,8,0,0,1,8-8H216A8,8,0,0,1,224,56Z', 'M216,40H40A16,16,0,0,0,24,56V200a16,16,0,0,0,16,16H216a16,16,0,0,0,16-16V56A16,16,0,0,0,216,40Zm0,16V158.75l-26.07-26.06a16,16,0,0,0-22.63,0l-20,20-44-44a16,16,0,0,0-22.62,0L40,149.37V56ZM40,172l52-52,80,80H40Zm176,28H194.63l-36-36,20-20L216,181.38V200ZM144,100a12,12,0,1,1,12,12A12,12,0,0,1,144,100Z'],
  folder: ['M128,80H32V56a8,8,0,0,1,8-8H92.69a8,8,0,0,1,5.65,2.34Z', 'M216,72H131.31L104,44.69A15.86,15.86,0,0,0,92.69,40H40A16,16,0,0,0,24,56V200.62A15.4,15.4,0,0,0,39.38,216H216.89A15.13,15.13,0,0,0,232,200.89V88A16,16,0,0,0,216,72ZM92.69,56l16,16H40V56ZM216,200H40V88H216Z']
};
function Icon({
  name,
  size = 16,
  label,
  style,
  ...rest
}) {
  const g = PHOSPHOR_DUOTONE[name];
  if (!g) return null;
  return /*#__PURE__*/React.createElement("svg", _extends({
    width: size,
    height: size,
    viewBox: "0 0 256 256",
    fill: "currentColor",
    role: label ? 'img' : undefined,
    "aria-label": label,
    "aria-hidden": label ? undefined : true,
    style: {
      display: 'block',
      flex: 'none',
      ...style
    }
  }, rest), /*#__PURE__*/React.createElement("path", {
    opacity: "0.25",
    d: g[0]
  }), /*#__PURE__*/React.createElement("path", {
    d: g[1]
  }));
}
Object.assign(__ds_scope, { PHOSPHOR_DUOTONE, Icon });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/icons/Icon.jsx", error: String((e && e.message) || e) }); }

// components/navigation/Nav.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Nav({
  brand,
  links = [],
  action,
  className,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("nav", _extends({
    className: ['nav', className].filter(Boolean).join(' ')
  }, rest), /*#__PURE__*/React.createElement("span", {
    className: "nav-brand"
  }, brand), links.map((l, i) => /*#__PURE__*/React.createElement("a", {
    key: i,
    href: l.href || '#',
    "aria-current": l.current ? 'page' : undefined,
    onClick: l.onClick
  }, l.label)), action);
}
Object.assign(__ds_scope, { Nav });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/navigation/Nav.jsx", error: String((e && e.message) || e) }); }

// components/print/Cmyk.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
// The showcase treatment: a photograph printed as its four misregistered process plates. Requires the filter defs from assets/print-plates.js in the document (loaded once here if missing).
function Cmyk({
  src,
  alt = '',
  aspectRatio = '3 / 2',
  mode = 'compound',
  platesSrc,
  className,
  style,
  ...rest
}) {
  React.useEffect(() => {
    if (document.getElementById('broadsheet-print-plates') || document.querySelector('script[data-print-plates]')) return;
    const s = document.createElement('script');
    s.src = platesSrc || (window.__broadsheetBase ? window.__broadsheetBase + '/assets/print-plates.js' : 'assets/print-plates.js');
    s.setAttribute('data-print-plates', '');
    document.head.appendChild(s);
  }, [platesSrc]);
  const cls = ['cmyk', className].filter(Boolean).join(' ');
  if (mode === 'plates') {
    return /*#__PURE__*/React.createElement("figure", _extends({
      className: cls,
      style: style
    }, rest), /*#__PURE__*/React.createElement("img", {
      src: src,
      alt: alt,
      style: {
        width: '100%',
        aspectRatio,
        objectFit: 'cover'
      }
    }), /*#__PURE__*/React.createElement("img", {
      className: "sep-c",
      src: src,
      alt: "",
      "aria-hidden": "true"
    }), /*#__PURE__*/React.createElement("img", {
      className: "sep-m",
      src: src,
      alt: "",
      "aria-hidden": "true"
    }), /*#__PURE__*/React.createElement("img", {
      className: "sep-y",
      src: src,
      alt: "",
      "aria-hidden": "true"
    }), /*#__PURE__*/React.createElement("img", {
      className: "sep-k",
      src: src,
      alt: "",
      "aria-hidden": "true"
    }));
  }
  return /*#__PURE__*/React.createElement("figure", _extends({
    className: cls,
    style: {
      overflow: 'visible',
      ...style
    }
  }, rest), /*#__PURE__*/React.createElement("div", {
    className: "print",
    style: {
      aspectRatio
    }
  }, /*#__PURE__*/React.createElement("img", {
    src: src,
    alt: alt
  })));
}
Object.assign(__ds_scope, { Cmyk });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/print/Cmyk.jsx", error: String((e && e.message) || e) }); }

// components/print/Halftone.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
// Newsprint dot screen for interface imagery.
function Halftone({
  src,
  alt = '',
  style,
  className,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("div", _extends({
    className: ['halftone', className].filter(Boolean).join(' '),
    style: style
  }, rest), /*#__PURE__*/React.createElement("img", {
    src: src,
    alt: alt,
    style: {
      width: '100%',
      height: '100%',
      objectFit: 'cover'
    }
  }));
}
Object.assign(__ds_scope, { Halftone });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/print/Halftone.jsx", error: String((e && e.message) || e) }); }

// components/print/HeadRule.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
// Front-page head furniture: the thick-thin head pair or the single cut rule — full-strength ink, the one place rules print.
function HeadRule({
  variant = 'head',
  className,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("hr", _extends({
    className: [variant === 'cut' ? 'rule-cut' : 'rule-head', className].filter(Boolean).join(' ')
  }, rest));
}
function Dateline({
  items = [],
  className,
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("p", _extends({
    className: className,
    style: {
      display: 'flex',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: '14px 28px',
      margin: 0,
      padding: '14px 0',
      fontSize: 13,
      lineHeight: '14px',
      letterSpacing: '0.08em',
      textTransform: 'uppercase',
      fontFeatureSettings: '"pnum" 1',
      color: 'var(--text-label)',
      ...style
    }
  }, rest), items.map((t, i) => /*#__PURE__*/React.createElement("span", {
    key: i
  }, t)));
}
Object.assign(__ds_scope, { HeadRule, Dateline });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/print/HeadRule.jsx", error: String((e && e.message) || e) }); }

// components/print/PlateText.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
// Display text set as three misregistered process plates (C/M/Y) — dark core from the multiply overlap, fringes from the registration drift.
function PlateText({
  children,
  variant = 'num',
  as = 'span',
  className,
  style,
  ...rest
}) {
  const Tag = as;
  const cls = [variant === 'head' ? 'cmyk-head' : 'cmyk-num', className].filter(Boolean).join(' ');
  return /*#__PURE__*/React.createElement(Tag, _extends({
    className: cls,
    style: style
  }, rest), /*#__PURE__*/React.createElement("span", {
    className: "paper"
  }, children), /*#__PURE__*/React.createElement("span", {
    className: "plate plate-c",
    "aria-hidden": "true"
  }, children), /*#__PURE__*/React.createElement("span", {
    className: "plate plate-m",
    "aria-hidden": "true"
  }, children), /*#__PURE__*/React.createElement("span", {
    className: "plate plate-y",
    "aria-hidden": "true"
  }, children));
}
Object.assign(__ds_scope, { PlateText });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/print/PlateText.jsx", error: String((e && e.message) || e) }); }

// components/surfaces/Card.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Card({
  kicker,
  title,
  body,
  meta,
  elevation,
  className,
  children,
  ...rest
}) {
  const cls = ['card', elevation && `elev-${elevation}`, className].filter(Boolean).join(' ');
  return /*#__PURE__*/React.createElement("div", _extends({
    className: cls
  }, rest), kicker && /*#__PURE__*/React.createElement("div", {
    className: "card-kicker"
  }, kicker), title && /*#__PURE__*/React.createElement("div", {
    className: "card-title"
  }, title), body && /*#__PURE__*/React.createElement("p", {
    className: "card-body"
  }, body), children, meta && /*#__PURE__*/React.createElement("div", {
    className: "card-meta"
  }, meta));
}
Object.assign(__ds_scope, { Card });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/surfaces/Card.jsx", error: String((e && e.message) || e) }); }

// components/surfaces/Dialog.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Dialog({
  open = true,
  title,
  children,
  actions,
  onClose,
  inline = false,
  id = 'dialog-title',
  ...rest
}) {
  if (!open) return null;
  return /*#__PURE__*/React.createElement("div", {
    className: "dialog-backdrop",
    style: inline ? {
      position: 'absolute'
    } : undefined,
    onClick: e => {
      if (e.target === e.currentTarget && onClose) onClose();
    }
  }, /*#__PURE__*/React.createElement("div", _extends({
    className: "dialog",
    role: "dialog",
    "aria-modal": "true",
    "aria-labelledby": id
  }, rest), title && /*#__PURE__*/React.createElement("div", {
    className: "dialog-title",
    id: id
  }, title), children && /*#__PURE__*/React.createElement("div", {
    className: "dialog-body"
  }, children), actions && /*#__PURE__*/React.createElement("div", {
    className: "dialog-actions"
  }, actions)));
}
Object.assign(__ds_scope, { Dialog });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/surfaces/Dialog.jsx", error: String((e && e.message) || e) }); }

// ui_kits/daybreak/Issues.jsx
try { (() => {
const {
  Table,
  Tag,
  Card,
  Icon,
  Button,
  HeadRule,
  Dateline,
  Halftone
} = window.BroadsheetDesignSystem_b6f617;
const ISSUES = [{
  n: '4,218',
  date: 'Sat 19 Sep',
  lead: 'The harbour reopens, quietly',
  words: 412,
  status: 'Final',
  tone: 'accent'
}, {
  n: '4,217',
  date: 'Fri 18 Sep',
  lead: 'Three councils, one bridge',
  words: 396,
  status: 'Final',
  tone: 'accent'
}, {
  n: '4,216',
  date: 'Thu 17 Sep',
  lead: 'A correction, and the rain',
  words: 388,
  status: 'Corrected',
  tone: 'accent-2'
}, {
  n: '4,215',
  date: 'Wed 16 Sep',
  lead: 'Nothing happened. Short issue.',
  words: 140,
  status: 'Final',
  tone: 'accent'
}, {
  n: '4,214',
  date: 'Tue 15 Sep',
  lead: 'The 7:12, on time for once',
  words: 401,
  status: 'Archived',
  tone: 'neutral'
}];
function Issues({
  go
}) {
  const [open, setOpen] = React.useState(ISSUES[0]);
  return /*#__PURE__*/React.createElement("div", {
    className: "wrap page"
  }, /*#__PURE__*/React.createElement("span", {
    className: "kicker"
  }, "Back issues"), /*#__PURE__*/React.createElement("h1", {
    className: "h-page"
  }, "The archive keeps both, honestly"), /*#__PURE__*/React.createElement("p", {
    className: "sub"
  }, "Every edition since \u2116 1, corrections at the top where they ran. Pick a morning."), /*#__PURE__*/React.createElement(HeadRule, null), /*#__PURE__*/React.createElement(Dateline, {
    items: ['Five most recent', 'All editions final at press', `${ISSUES.length} shown`]
  }), /*#__PURE__*/React.createElement(HeadRule, {
    variant: "cut"
  }), /*#__PURE__*/React.createElement("div", {
    className: "two"
  }, /*#__PURE__*/React.createElement(Table, {
    columns: [{
      key: 'n',
      label: '№'
    }, {
      key: 'date',
      label: 'Morning',
      muted: true
    }, {
      key: 'lead',
      label: 'Lead story',
      render: r => /*#__PURE__*/React.createElement("a", {
        href: "#",
        onClick: e => {
          e.preventDefault();
          setOpen(r);
        },
        style: {
          color: open === r ? 'var(--color-accent-700)' : 'inherit',
          textDecoration: 'none'
        }
      }, r.lead)
    }, {
      key: 'words',
      label: 'Words',
      align: 'right'
    }, {
      key: 'status',
      label: 'Status',
      render: r => /*#__PURE__*/React.createElement(Tag, {
        tone: r.tone
      }, r.status)
    }],
    rows: ISSUES
  }), /*#__PURE__*/React.createElement(Card, {
    kicker: `Issue № ${open.n}`,
    title: open.lead,
    body: `${open.words} words · ${open.date}. Three editors read everything published overnight, argued until five, and printed what survived.`,
    meta: /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Icon, {
      name: "calendar",
      size: 13
    }), /*#__PURE__*/React.createElement("span", null, "Delivered 6:00"))
  }, /*#__PURE__*/React.createElement(Halftone, {
    src: "../../assets/photo.jpg",
    alt: "",
    style: {
      aspectRatio: '3 / 2',
      borderRadius: 'var(--radius-md)'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 'var(--space-2)'
    }
  }, /*#__PURE__*/React.createElement(Button, null, "Read it", /*#__PURE__*/React.createElement(Icon, {
    name: "arrow",
    size: 14
  })), /*#__PURE__*/React.createElement(Button, {
    variant: "ghost",
    onClick: () => go('landing')
  }, "Back to today\u2019s")))));
}
Object.assign(window, {
  Issues
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/daybreak/Issues.jsx", error: String((e && e.message) || e) }); }

// ui_kits/daybreak/Landing.jsx
try { (() => {
const DS = window.BroadsheetDesignSystem_b6f617;
const {
  Nav,
  Button,
  HeadRule,
  Dateline,
  Cmyk,
  Input,
  PlateText,
  Dialog
} = DS;
function Landing({
  go,
  onSubscribe
}) {
  const [email, setEmail] = React.useState('');
  const [done, setDone] = React.useState(false);
  const submit = () => {
    if (!email.includes('@')) return;
    setDone(true);
    onSubscribe && onSubscribe(email);
  };
  const Head = ({
    children
  }) => /*#__PURE__*/React.createElement(PlateText, {
    variant: "head",
    as: "span",
    className: "line",
    style: {
      display: 'block'
    }
  }, children);
  return /*#__PURE__*/React.createElement("div", {
    className: "wrap"
  }, /*#__PURE__*/React.createElement("section", {
    className: "hero"
  }, /*#__PURE__*/React.createElement("h1", {
    className: "display"
  }, /*#__PURE__*/React.createElement(Head, null, "All the news."), /*#__PURE__*/React.createElement(Head, null, "None of the feed.")), /*#__PURE__*/React.createElement("p", {
    className: "sub"
  }, "Daybreak is the five-minute morning digest: every story that matters, four hundred words, edited by people who went to bed angry about adjectives. On paper-white, in your inbox by six."), /*#__PURE__*/React.createElement("div", {
    className: "row"
  }, /*#__PURE__*/React.createElement(Button, {
    onClick: () => go('subscribe')
  }, "Subscribe"), /*#__PURE__*/React.createElement(Button, {
    variant: "ghost",
    onClick: () => go('issues')
  }, "Read a back issue"))), /*#__PURE__*/React.createElement("section", {
    className: "frontpage",
    "aria-label": "Daybreak, by the numbers"
  }, /*#__PURE__*/React.createElement(HeadRule, null), /*#__PURE__*/React.createElement(Dateline, {
    items: ['№ 4,218', 'Printed overnight', 'Fair, then clearing', 'Price: one coffee']
  }), /*#__PURE__*/React.createElement(HeadRule, {
    variant: "cut"
  }), /*#__PURE__*/React.createElement("div", {
    className: "index"
  }, /*#__PURE__*/React.createElement("p", {
    className: "ix"
  }, /*#__PURE__*/React.createElement("span", null, "On the doorstep, daily"), /*#__PURE__*/React.createElement("span", {
    className: "ix-leader"
  }), /*#__PURE__*/React.createElement("span", {
    className: "ix-num spot"
  }, "6:00")), /*#__PURE__*/React.createElement("p", {
    className: "ix"
  }, /*#__PURE__*/React.createElement("span", null, "Words \u2014 the whole morning"), /*#__PURE__*/React.createElement("span", {
    className: "ix-leader"
  }), /*#__PURE__*/React.createElement("span", {
    className: "ix-num"
  }, "400")), /*#__PURE__*/React.createElement("p", {
    className: "ix"
  }, /*#__PURE__*/React.createElement("span", null, "Push notifications, ever"), /*#__PURE__*/React.createElement("span", {
    className: "ix-leader"
  }), /*#__PURE__*/React.createElement("span", {
    className: "ix-num"
  }, "0")), /*#__PURE__*/React.createElement("p", {
    className: "ix"
  }, /*#__PURE__*/React.createElement("span", null, "Edition, final at press"), /*#__PURE__*/React.createElement("span", {
    className: "ix-leader"
  }), /*#__PURE__*/React.createElement("span", {
    className: "ix-num"
  }, "1"))), /*#__PURE__*/React.createElement(HeadRule, {
    variant: "cut"
  })), /*#__PURE__*/React.createElement("section", {
    className: "features",
    id: "inside"
  }, /*#__PURE__*/React.createElement("span", {
    className: "kicker"
  }, "Inside every issue"), /*#__PURE__*/React.createElement("div", {
    className: "cols"
  }, /*#__PURE__*/React.createElement("div", {
    className: "col"
  }, /*#__PURE__*/React.createElement("h2", null, "Edited, not aggregated"), /*#__PURE__*/React.createElement("p", null, "No algorithm chooses your morning. Three editors read everything published overnight, argue until five, and print what survived. If nothing happened, the issue is short \u2014 we consider that a feature of the news, not a failure of the product.")), /*#__PURE__*/React.createElement("div", {
    className: "col"
  }, /*#__PURE__*/React.createElement("h2", null, "The fold, respected"), /*#__PURE__*/React.createElement("p", null, "The lead story is the lead story. Everything above the fold matters today; everything below it can wait for lunch. Nothing is promoted because you lingered on it \u2014 lingering is between you and your coffee.")), /*#__PURE__*/React.createElement("div", {
    className: "col"
  }, /*#__PURE__*/React.createElement("h2", null, "Corrections, printed"), /*#__PURE__*/React.createElement("p", null, "When we get it wrong, the correction runs at the top of the next issue, set in the same size as the mistake. The record matters more than the streak, and the archive keeps both honestly.")))), /*#__PURE__*/React.createElement("section", {
    className: "split",
    id: "print"
  }, /*#__PURE__*/React.createElement("div", {
    className: "split-copy"
  }, /*#__PURE__*/React.createElement("span", {
    className: "kicker"
  }, "In print"), /*#__PURE__*/React.createElement("h2", {
    className: "split-title"
  }, "Printed in four inks"), /*#__PURE__*/React.createElement("p", {
    className: "note"
  }, "Every photograph prints as its four process plates \u2014 cyan, magenta, yellow and black, a breath out of register, the way the presses have made pictures for a century. It reads at any size, and it reminds you the morning was made by hand.")), /*#__PURE__*/React.createElement(Cmyk, {
    className: "split-figure",
    src: "../../assets/photo-color.jpg",
    aspectRatio: "986 / 660",
    alt: "Product photograph printed as its four process plates, slightly out of register"
  })), /*#__PURE__*/React.createElement("section", {
    className: "quote"
  }, /*#__PURE__*/React.createElement("figure", null, /*#__PURE__*/React.createElement("blockquote", null, "\u201CI cancelled four subscriptions and three anxieties. The 7:12 to Paddington is long enough for the whole paper.\u201D"), /*#__PURE__*/React.createElement("figcaption", null, "\u2014 M. Okafor, reads it on the train"))), /*#__PURE__*/React.createElement("section", {
    className: "close",
    id: "subscribe"
  }, /*#__PURE__*/React.createElement("h3", null, "Tomorrow\u2019s issue is already being argued about"), /*#__PURE__*/React.createElement("p", {
    className: "sub"
  }, "Free for your first month of mornings. After that, the price of one coffee \u2014 the paper pairs well with it."), /*#__PURE__*/React.createElement("div", {
    className: "signup"
  }, /*#__PURE__*/React.createElement(Input, {
    type: "email",
    placeholder: "you@example.com",
    "aria-label": "Email address",
    value: email,
    onChange: e => setEmail(e.target.value),
    onKeyDown: e => e.key === 'Enter' && submit()
  }), /*#__PURE__*/React.createElement(Button, {
    onClick: submit
  }, "Subscribe"))), /*#__PURE__*/React.createElement("footer", null, "Daybreak is printed overnight and delivered by six. Set in Source Serif 4."), /*#__PURE__*/React.createElement(Dialog, {
    open: done,
    title: "You\u2019re on the list",
    onClose: () => setDone(false),
    actions: /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Button, {
      variant: "secondary",
      onClick: () => setDone(false)
    }, "Close"), /*#__PURE__*/React.createElement(Button, {
      onClick: () => {
        setDone(false);
        go('preferences');
      }
    }, "Set preferences"))
  }, "Tomorrow\u2019s edition goes to ", email, " at six. Nothing else, ever, unless you ask."));
}
Object.assign(window, {
  Landing
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/daybreak/Landing.jsx", error: String((e && e.message) || e) }); }

// ui_kits/daybreak/Preferences.jsx
try { (() => {
const {
  Field,
  Input,
  Segmented,
  RadioGroup,
  Button,
  Icon,
  Dialog
} = window.BroadsheetDesignSystem_b6f617;
function Preferences({
  go
}) {
  const [saved, setSaved] = React.useState(false);
  const [confirm, setConfirm] = React.useState(false);
  return /*#__PURE__*/React.createElement("div", {
    className: "wrap page"
  }, /*#__PURE__*/React.createElement("span", {
    className: "kicker"
  }, "Preferences"), /*#__PURE__*/React.createElement("h1", {
    className: "h-page"
  }, "How the paper arrives"), /*#__PURE__*/React.createElement("p", {
    className: "sub"
  }, "Delivery, format and what to do when we get it wrong. Nothing here is remembered about what you read."), /*#__PURE__*/React.createElement("div", {
    className: "two form"
  }, /*#__PURE__*/React.createElement("div", {
    className: "stack"
  }, /*#__PURE__*/React.createElement(Field, {
    label: "Name on the masthead",
    htmlFor: "pn"
  }, /*#__PURE__*/React.createElement(Input, {
    id: "pn",
    defaultValue: "M. Okafor"
  })), /*#__PURE__*/React.createElement(Field, {
    label: "Email",
    htmlFor: "pe"
  }, /*#__PURE__*/React.createElement(Input, {
    id: "pe",
    type: "email",
    defaultValue: "okafor@example.com"
  })), /*#__PURE__*/React.createElement(Field, {
    label: "Note to the editors",
    htmlFor: "pt"
  }, /*#__PURE__*/React.createElement(Input, {
    id: "pt",
    multiline: true,
    rows: 3,
    placeholder: "Fewer adjectives, please."
  }))), /*#__PURE__*/React.createElement("div", {
    className: "stack"
  }, /*#__PURE__*/React.createElement(Field, {
    label: "Edition",
    labelId: "ed"
  }, /*#__PURE__*/React.createElement(Segmented, {
    name: "edition",
    "aria-labelledby": "ed",
    options: [{
      value: 'morning',
      label: 'Morning',
      icon: /*#__PURE__*/React.createElement(Icon, {
        name: "sparkle",
        size: 13
      })
    }, {
      value: 'weekend',
      label: 'Weekend',
      icon: /*#__PURE__*/React.createElement(Icon, {
        name: "calendar",
        size: 13
      })
    }, {
      value: 'print',
      label: 'Print',
      icon: /*#__PURE__*/React.createElement(Icon, {
        name: "image",
        size: 13
      })
    }]
  })), /*#__PURE__*/React.createElement(RadioGroup, {
    label: "When a story is corrected",
    name: "corr",
    options: [{
      value: 'top',
      label: 'Run it at the top of the next issue'
    }, {
      value: 'note',
      label: 'Send a separate note'
    }, {
      value: 'quiet',
      label: 'Fix the archive quietly'
    }]
  }), /*#__PURE__*/React.createElement(RadioGroup, {
    label: "Push notifications",
    name: "push",
    options: [{
      value: 'never',
      label: 'Never — this is a newspaper'
    }]
  }))), /*#__PURE__*/React.createElement("div", {
    className: "actions"
  }, /*#__PURE__*/React.createElement(Button, {
    variant: "secondary",
    onClick: () => setConfirm(true)
  }, "Cancel subscription"), /*#__PURE__*/React.createElement(Button, {
    onClick: () => setSaved(true)
  }, "Save changes"), saved && /*#__PURE__*/React.createElement("span", {
    className: "saved"
  }, "Saved. Tomorrow\u2019s edition follows the new settings.")), /*#__PURE__*/React.createElement(Dialog, {
    open: confirm,
    title: "Stop the paper?",
    onClose: () => setConfirm(false),
    actions: /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Button, {
      variant: "secondary",
      onClick: () => setConfirm(false)
    }, "Keep reading"), /*#__PURE__*/React.createElement(Button, {
      onClick: () => {
        setConfirm(false);
        go('landing');
      }
    }, "Cancel it"))
  }, "Your archive stays where it is. Tomorrow\u2019s edition will not arrive, and no one will write to ask why."));
}
Object.assign(window, {
  Preferences
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/daybreak/Preferences.jsx", error: String((e && e.message) || e) }); }

__ds_ns.Button = __ds_scope.Button;

__ds_ns.Tag = __ds_scope.Tag;

__ds_ns.Table = __ds_scope.Table;

__ds_ns.Field = __ds_scope.Field;

__ds_ns.Input = __ds_scope.Input;

__ds_ns.Radio = __ds_scope.Radio;

__ds_ns.RadioGroup = __ds_scope.RadioGroup;

__ds_ns.Segmented = __ds_scope.Segmented;

__ds_ns.PHOSPHOR_DUOTONE = __ds_scope.PHOSPHOR_DUOTONE;

__ds_ns.Icon = __ds_scope.Icon;

__ds_ns.Nav = __ds_scope.Nav;

__ds_ns.Cmyk = __ds_scope.Cmyk;

__ds_ns.Halftone = __ds_scope.Halftone;

__ds_ns.HeadRule = __ds_scope.HeadRule;

__ds_ns.Dateline = __ds_scope.Dateline;

__ds_ns.PlateText = __ds_scope.PlateText;

__ds_ns.Card = __ds_scope.Card;

__ds_ns.Dialog = __ds_scope.Dialog;

})();
