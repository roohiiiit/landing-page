#!/usr/bin/env python3
"""
Trace signature.png into:
  1. signature-fill.svg       (filled vector via potrace)
  2. signature-centerline.svg (ordered, smoothed pen-stroke paths via skeletonize)

Both SVGs share the exact same tight viewBox so signature-centerline.svg
can act as an animated SVG <mask> over signature-fill.svg.
"""

import math
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.measure import approximate_polygon
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent.parent
INPUT_PNG = ROOT / "signature.png"
FILL_SVG = ROOT / "signature-fill.svg"
CENTERLINE_SVG = ROOT / "signature-centerline.svg"

THRESHOLD = 140      # grayscale threshold (0..255); pixels < THRESHOLD are ink
PAD = 24             # padding around tight ink bounding box (px)
SPUR_MAX_LEN = 10    # prune tiny skeleton spurs attached to junctions (px)
RDP_TOLERANCE = 1.8  # Ramer-Douglas-Peucker simplification tolerance (px)
MIN_STROKE_LEN = 4   # discard tiny 1-3px junction artifacts (unless isolated dot)


def load_ink_mask(path: Path) -> tuple[np.ndarray, int, int, float]:
    """Load PNG, composite onto white, threshold, crop to ink bbox + PAD."""
    im = Image.open(path).convert("RGBA")
    arr = np.array(im, dtype=np.float32)
    alpha = arr[..., 3:4] / 255.0
    rgb = arr[..., :3] * alpha + 255.0 * (1.0 - alpha)
    gray = rgb.mean(axis=2)
    mask = gray < THRESHOLD

    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("No ink pixels found in signature.png")

    h_full, w_full = mask.shape
    x0 = max(0, int(xs.min()) - PAD)
    y0 = max(0, int(ys.min()) - PAD)
    x1 = min(w_full, int(xs.max()) + PAD + 1)
    y1 = min(h_full, int(ys.max()) + PAD + 1)

    cropped = mask[y0:y1, x0:x1]
    crop_h, crop_w = cropped.shape

    dist = distance_transform_edt(cropped)
    max_stroke_width = float(dist.max() * 2.0)
    return cropped, crop_w, crop_h, max_stroke_width


def generate_fill_svg(mask: np.ndarray, w: int, h: int, out_path: Path) -> None:
    """Use potrace to create a clean filled vector SVG."""
    img = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), mode="L")

    with tempfile.TemporaryDirectory() as tmpdir:
        bmp_path = os.path.join(tmpdir, "mask.bmp")
        raw_svg = os.path.join(tmpdir, "raw.svg")
        img.save(bmp_path)

        subprocess.run(
            [
                "potrace",
                bmp_path,
                "-s",
                "-o",
                raw_svg,
                "--turdsize",
                "2",
                "--alphamax",
                "1.0",
                "--opttolerance",
                "0.2",
            ],
            check=True,
        )

        raw_text = Path(raw_svg).read_text(encoding="utf-8")

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    root = ET.fromstring(raw_text)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    g = root.find("svg:g", ns)
    if g is None:
        g = root.find("g")

    transform = g.attrib.get("transform", "") if g is not None else ""
    paths = g.findall("svg:path", ns) if g is not None else root.findall(".//svg:path", ns)
    path_ds = [p.attrib["d"] for p in paths if "d" in p.attrib]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'  <g transform="{transform}" fill="currentColor" stroke="none">',
    ]
    for d in path_ds:
        d_clean = re.sub(r"\s+", " ", d).strip()
        lines.append(f'    <path d="{d_clean}"/>')
    lines.append("  </g>")
    lines.append("</svg>")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def neighbors_8(y: int, x: int, skel: np.ndarray) -> list[tuple[int, int]]:
    h, w = skel.shape
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                out.append((ny, nx))
    return out


def prune_spurs(skel: np.ndarray, max_len: int) -> np.ndarray:
    """Remove short degree-1 branches attached to junctions."""
    skel = skel.copy()
    changed = True
    while changed:
        changed = False
        ys, xs = np.where(skel)
        for y, x in zip(ys, xs):
            if not skel[y, x]:
                continue
            nbs = neighbors_8(y, x, skel)
            if len(nbs) == 1:
                branch = [(y, x)]
                prev = (y, x)
                curr = nbs[0]
                hit_junction = False
                while len(branch) <= max_len:
                    cnbs = [p for p in neighbors_8(curr[0], curr[1], skel) if p != prev]
                    if len(cnbs) == 0:
                        break
                    if len(cnbs) >= 2:
                        hit_junction = True
                        break
                    branch.append(curr)
                    prev = curr
                    curr = cnbs[0]
                if hit_junction and len(branch) <= max_len:
                    for by, bx in branch:
                        skel[by, bx] = False
                    changed = True
    return skel


def extract_continuous_strokes(skel: np.ndarray) -> list[list[tuple[float, float]]]:
    """
    Walk skeleton pixels into continuous strokes.
    At crossings/junctions, continue along the straightest outgoing direction
    (minimum turning angle) so intersecting pen strokes stay intact.
    """
    ys, xs = np.where(skel)
    nodes = set(zip(ys.tolist(), xs.tolist()))
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {
        p: neighbors_8(p[0], p[1], skel) for p in nodes
    }

    visited_nodes: set[tuple[int, int]] = set()
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    strokes: list[list[tuple[float, float]]] = []

    def mark_edge(a: tuple[int, int], b: tuple[int, int]) -> None:
        visited_edges.add((a, b))
        visited_edges.add((b, a))

    def pick_straightest(
        chain: list[tuple[int, int]], cands: list[tuple[int, int]]
    ) -> tuple[int, int] | None:
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        # Compute incoming tangent over last up to 6 pixels
        k = min(6, len(chain) - 1)
        p_back = chain[-1 - k]
        p_cur = chain[-1]
        vx = p_cur[1] - p_back[1]
        vy = p_cur[0] - p_back[0]
        v_norm = math.hypot(vx, vy) or 1.0
        ux, uy = vx / v_norm, vy / v_norm

        best_p = None
        best_dot = -2.0
        for c in cands:
            cx = c[1] - p_cur[1]
            cy = c[0] - p_cur[0]
            cn = math.hypot(cx, cy) or 1.0
            dot = (cx / cn) * ux + (cy / cn) * uy
            if dot > best_dot:
                best_dot = dot
                best_p = c
        # Only continue through a junction if the turn is genuinely straight (< ~45 deg)
        if best_dot > 0.68:
            return best_p
        return None

    def walk_from(start: tuple[int, int], first_next: tuple[int, int]) -> list[tuple[int, int]]:
        chain = [start, first_next]
        mark_edge(start, first_next)
        visited_nodes.add(start)
        visited_nodes.add(first_next)
        curr = first_next
        while True:
            cands = [p for p in adj[curr] if (curr, p) not in visited_edges]
            # Prefer unvisited nodes at junctions so we don't loop on 1px triangles
            unvisited_cands = [p for p in cands if p not in visited_nodes]
            pool = unvisited_cands if unvisited_cands else cands
            if not pool:
                break
            if len(pool) == 1 and len(adj[curr]) <= 2:
                nxt = pool[0]
            else:
                nxt = pick_straightest(chain, pool)
                if nxt is None:
                    break
            chain.append(nxt)
            mark_edge(curr, nxt)
            visited_nodes.add(nxt)
            curr = nxt
        return chain

    # 1. Start from all true endpoints (degree == 1), ordered left-to-right, top-to-bottom
    endpoints = sorted([p for p, nbs in adj.items() if len(nbs) == 1], key=lambda p: (p[1], p[0]))
    for ep in endpoints:
        for nxt in adj[ep]:
            if (ep, nxt) not in visited_edges:
                chain = walk_from(ep, nxt)
                strokes.append([(float(x), float(y)) for y, x in chain])

    # 2. Pick up any remaining unvisited branches / loops from junctions or degree-2 nodes
    for start in sorted(nodes, key=lambda p: (p[1], p[0])):
        for nxt in adj[start]:
            if (start, nxt) not in visited_edges:
                chain = walk_from(start, nxt)
                if len(chain) >= MIN_STROKE_LEN:
                    strokes.append([(float(x), float(y)) for y, x in chain])

    # 3. Preserve isolated 1-pixel dots (degree == 0)
    for p, nbs in adj.items():
        if len(nbs) == 0:
            strokes.append([(float(p[1]), float(p[0])), (float(p[1]) + 0.5, float(p[0]))])

    return strokes


def orient_and_order_strokes(
    strokes: list[list[tuple[float, float]]]
) -> list[list[tuple[float, float]]]:
    """Order strokes left-to-right and in a natural pen-flow order."""
    if not strokes:
        return []

    oriented = []
    for pts in strokes:
        dx = abs(pts[-1][0] - pts[0][0])
        dy = abs(pts[-1][1] - pts[0][1])
        if dx >= dy:
            if pts[0][0] > pts[-1][0]:
                pts = list(reversed(pts))
        else:
            if pts[0][1] > pts[-1][1]:
                pts = list(reversed(pts))
        oriented.append(pts)

    remaining = list(oriented)
    remaining.sort(key=lambda pts: (min(p[0] for p in pts), min(p[1] for p in pts)))

    ordered = [remaining.pop(0)]
    while remaining:
        cur_end = ordered[-1][-1]
        best_idx = 0
        best_score = float("inf")
        best_rev = False

        for i, pts in enumerate(remaining):
            d_start = math.hypot(pts[0][0] - cur_end[0], pts[0][1] - cur_end[1])
            d_end = math.hypot(pts[-1][0] - cur_end[0], pts[-1][1] - cur_end[1])
            min_x = min(p[0] for p in pts)

            for dist, rev in ((d_start, False), (d_end, True)):
                score = dist if dist < 35.0 else dist * 0.35 + min_x * 1.3
                if score < best_score:
                    best_score = score
                    best_idx = i
                    best_rev = rev

        nxt = remaining.pop(best_idx)
        if best_rev:
            nxt = list(reversed(nxt))
        ordered.append(nxt)

    return ordered


def polyline_to_bezier_path(pts: list[tuple[float, float]]) -> str:
    """Simplify polyline with RDP and convert to smooth Catmull-Rom cubic Beziers."""
    arr = np.array(pts, dtype=np.float64)
    if len(arr) >= 3:
        simp = approximate_polygon(arr, tolerance=RDP_TOLERANCE)
        if len(simp) < 2:
            simp = arr[[0, -1]]
    else:
        simp = arr

    if len(simp) == 1:
        x, y = simp[0]
        return f"M {x:.2f} {y:.2f} L {x + 0.5:.2f} {y:.2f}"

    if len(simp) == 2:
        x0, y0 = simp[0]
        x1, y1 = simp[1]
        return f"M {x0:.2f} {y0:.2f} L {x1:.2f} {y1:.2f}"

    d = [f"M {simp[0][0]:.2f} {simp[0][1]:.2f}"]
    n = len(simp)
    for i in range(n - 1):
        p0 = simp[i - 1] if i > 0 else simp[0]
        p1 = simp[i]
        p2 = simp[i + 1]
        p3 = simp[i + 2] if i + 2 < n else simp[-1]

        c1x = p1[0] + (p2[0] - p0[0]) / 6.0
        c1y = p1[1] + (p2[1] - p0[1]) / 6.0
        c2x = p2[0] - (p3[0] - p1[0]) / 6.0
        c2y = p2[1] - (p3[1] - p1[1]) / 6.0

        d.append(
            f"C {c1x:.2f} {c1y:.2f}, {c2x:.2f} {c2y:.2f}, {p2[0]:.2f} {p2[1]:.2f}"
        )
    return " ".join(d)


def generate_centerline_svg(
    mask: np.ndarray, w: int, h: int, max_stroke_width: float, out_path: Path
) -> int:
    """Skeletonize mask, trace ordered polylines, and save smooth Bezier SVG."""
    skel = skeletonize(mask)
    skel = prune_spurs(skel, SPUR_MAX_LEN)

    raw_strokes = extract_continuous_strokes(skel)
    ordered = orient_and_order_strokes(raw_strokes)

    paths_d = [polyline_to_bezier_path(pts) for pts in ordered if len(pts) >= 2]
    mask_stroke = round(max(max_stroke_width * 1.4, 20.0), 1)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'  <g fill="none" stroke="currentColor" stroke-width="{mask_stroke}" stroke-linecap="round" stroke-linejoin="round">',
    ]
    for d in paths_d:
        lines.append(f'    <path d="{d}"/>')
    lines.append("  </g>")
    lines.append("</svg>")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(paths_d)


def main() -> None:
    if not INPUT_PNG.exists():
        raise FileNotFoundError(f"Missing {INPUT_PNG}")

    mask, w, h, max_w = load_ink_mask(INPUT_PNG)
    generate_fill_svg(mask, w, h, FILL_SVG)
    n_strokes = generate_centerline_svg(mask, w, h, max_w, CENTERLINE_SVG)

    print(f"Traced {INPUT_PNG.name} -> viewBox 0 0 {w} {h}")
    print(f"  Max ink width: {max_w:.1f}px (mask stroke: {max(max_w * 1.4, 20.0):.1f}px)")
    print(f"  {FILL_SVG.name}: saved")
    print(f"  {CENTERLINE_SVG.name}: saved ({n_strokes} strokes)")


if __name__ == "__main__":
    main()
