"""
analyze_autolog.py — Analiza data/autolog.csv generado por collect_cil_data.py
y valida el comportamiento del seguidor de carril contra la geometría de
city_traffic_2025_01.wbt (extraída manualmente el 2026-06-23).

Uso:
    python code/analyze_autolog.py
    python code/analyze_autolog.py --log data/autolog.csv --plot

Salida:
    - Resumen por consola (oscillation, lane error, intersection detection)
    - data/autolog_report.txt
    - (con --plot) data/autolog_plots.png
"""

import csv, math, os, sys, argparse
from pathlib import Path

# ── Geometría del mundo (city_traffic_2025_01.wbt) ────────────────────────────
# Coordenadas en metros en el plano XY de Webots.

WORLD_INTERSECTIONS = [
    {"id": "25", "x":  45.0, "y": -45.0, "label": "Cruce SE"},
    {"id": "26", "x": -45.0, "y":  45.0, "label": "Cruce NO"},
    {"id": "27", "x": 105.0, "y":  93.0, "label": "Cruce NE"},
]

# Radio de las intersecciones (startRoadsLength=8.75, + la mitad de road width 21.5/2)
INTER_RADIUS_WBT = 20.0   # metros: dentro = zona de cruce real

WORLD_STRAIGHTS = [
    # (id, cx, cy, rot_rad, length)
    # rot=0 → corre a lo largo de X; rot=π/2 → corre a lo largo de Y
    ("1",   -105.0,    4.5, -math.pi/2,  69.0),
    ("3",    -64.5, -105.0,         0.0,  69.0),
    ("5",     45.0,    4.5, -math.pi/2,  30.0),
    ("7",    -25.5,   45.0,         0.0,  30.0),
    ("9",     -4.5,  -45.0,         0.0,  30.0),
    ("12",    85.5,   93.0,  math.pi,     9.0),
    ("13",    75.0, -187.5, -math.pi,   265.0),
    ("14",   165.0,   52.5, -math.pi/2, 150.0),
    ("15",    64.5,  231.0,  math.pi,   254.0),
    ("16",   105.0,  112.5,  math.pi/2,  78.0),
    ("17",   105.0,   -4.5,  math.pi/2,  78.0),
    ("23",   -45.0,   25.5, -math.pi/2,  30.0),
    ("24",   -45.0,   64.5,  math.pi/2,  69.0),
    ("26",  -230.0,  190.5, -math.pi/2, 337.5),
]

ROAD_WIDTH = 21.5   # todos los segmentos
BMW_SPAWN  = (-111.45, 16.11)


def dist2d(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def nearest_intersection(gx, gy):
    best, bd = None, float("inf")
    for inter in WORLD_INTERSECTIONS:
        d = dist2d(gx, gy, inter["x"], inter["y"])
        if d < bd:
            bd, best = d, inter
    return best, round(bd, 2)


def on_road_segment(gx, gy, seg_id, cx, cy, rot, length):
    """Retorna True si (gx,gy) está sobre el segmento recto (dentro del AABB)."""
    half_l = length / 2.0
    half_w = ROAD_WIDTH / 2.0
    # Transformar al espacio local del segmento
    dx, dy = gx - cx, gy - cy
    cos_r, sin_r = math.cos(-rot), math.sin(-rot)
    lx = cos_r * dx - sin_r * dy
    ly = sin_r * dx + cos_r * dy
    return abs(lx) <= half_l + 2.0 and abs(ly) <= half_w + 2.0


def classify_position(gx, gy):
    """Clasifica la posición GPS: 'intersection', 'straight:id', o 'unknown'."""
    _, d_inter = nearest_intersection(gx, gy)
    if d_inter < INTER_RADIUS_WBT:
        _, nearest = nearest_intersection(gx, gy), None
        for inter in WORLD_INTERSECTIONS:
            if dist2d(gx, gy, inter["x"], inter["y"]) == d_inter:
                return f"intersection:{inter['id']}:{inter['label']}"
    for seg_id, cx, cy, rot, length in WORLD_STRAIGHTS:
        if on_road_segment(gx, gy, seg_id, cx, cy, rot, length):
            return f"straight:{seg_id}"
    return "unknown"


def load_log(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({k: float(v) if v not in ("", "nan") else float("nan")
                              for k, v in row.items()})
            except Exception:
                pass
    return rows


def compute_oscillation(series, threshold=0.0):
    """Cuenta cambios de signo en una serie (oscilaciones)."""
    crossings = 0
    prev_sign = None
    for v in series:
        if math.isnan(v):
            continue
        sign = 1 if v > threshold else -1
        if prev_sign is not None and sign != prev_sign:
            crossings += 1
        prev_sign = sign
    return crossings


def analyze(log_path, plot=False):
    if not os.path.exists(log_path):
        print(f"[ERROR] Log no encontrado: {log_path}")
        sys.exit(1)

    rows = load_log(log_path)
    if not rows:
        print("[ERROR] Log vacío.")
        sys.exit(1)

    n = len(rows)
    duration_sim_s = (rows[-1]["sim_ms"] - rows[0]["sim_ms"]) / 1000.0
    duration_wall  = rows[-1]["wall_t"] - rows[0]["wall_t"]

    # ── Extraer series ────────────────────────────────────────────────────────
    error_norms    = [r.get("error_norm", float("nan"))    for r in rows]
    smooth_centers = [r.get("smooth_center", float("nan")) for r in rows]
    steers         = [r.get("steer", float("nan"))         for r in rows]
    confidences    = [r.get("confidence", float("nan"))    for r in rows]
    road_fracs     = [r.get("avg_road_frac", float("nan")) for r in rows]
    n_valid        = [r.get("n_valid_rows", float("nan"))  for r in rows]
    is_inters      = [r.get("is_intersection_algo", r.get("is_intersection", 0)) for r in rows]
    found_seeds    = [r.get("found_seed", 0)               for r in rows]
    gps_xs         = [r.get("gps_x", float("nan"))         for r in rows]
    gps_ys         = [r.get("gps_y", float("nan"))         for r in rows]
    states_raw     = rows  # for state-based breakdown

    # Filtrar NaN
    valid_errors = [v for v in error_norms  if not math.isnan(v)]
    valid_steers = [v for v in steers       if not math.isnan(v)]
    valid_confs  = [v for v in confidences  if not math.isnan(v)]
    valid_fracs  = [v for v in road_fracs   if not math.isnan(v)]

    # ── Oscilación ────────────────────────────────────────────────────────────
    error_crossings = compute_oscillation(valid_errors)
    steer_crossings = compute_oscillation(valid_steers)
    rate_crossings  = error_crossings / max(duration_sim_s, 1.0)

    # ── Lane error ────────────────────────────────────────────────────────────
    mean_error = sum(valid_errors) / len(valid_errors) if valid_errors else float("nan")
    rms_error  = math.sqrt(sum(v**2 for v in valid_errors) / len(valid_errors)) if valid_errors else float("nan")
    max_steer  = max(abs(v) for v in valid_steers) if valid_steers else 0.0
    mean_conf  = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0

    # ── Seed loss ─────────────────────────────────────────────────────────────
    seed_lost_pct = (1.0 - sum(found_seeds) / max(n, 1)) * 100.0

    # ── Road frac stats ───────────────────────────────────────────────────────
    mean_frac = sum(valid_fracs) / len(valid_fracs) if valid_fracs else 0.0
    inter_detected_pct = sum(1 for v in is_inters if v > 0.5) / max(n, 1) * 100.0

    # ── Clasificación GPS vs intersecciones del mundo ─────────────────────────
    inter_tp = 0   # is_intersection=True  AND GPS cerca de una intersección real
    inter_fp = 0   # is_intersection=True  AND GPS lejos (falso positivo)
    inter_fn = 0   # is_intersection=False AND GPS dentro del radio de intersección
    road_unknown = 0

    for r in rows:
        gx, gy = r["gps_x"], r["gps_y"]
        if math.isnan(gx) or math.isnan(gy):
            continue
        _, d_inter = nearest_intersection(gx, gy)
        algo_says_inter = r["is_intersection"] > 0.5
        gps_in_inter    = d_inter < INTER_RADIUS_WBT

        if algo_says_inter and gps_in_inter:
            inter_tp += 1
        elif algo_says_inter and not gps_in_inter:
            inter_fp += 1
        elif not algo_says_inter and gps_in_inter:
            inter_fn += 1

        seg = classify_position(gx, gy)
        if seg == "unknown":
            road_unknown += 1

    # ── Steer rate limiter saturation ─────────────────────────────────────────
    deltas = [r["steer_delta"] for r in rows if not math.isnan(r["steer_delta"])]
    from_params = 0.05  # RATE_LIMIT actual
    saturated_pct = sum(1 for d in deltas if abs(d) >= from_params * 0.99) / max(len(deltas), 1) * 100.0

    # ── Reporte ───────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 62)
    lines.append("  REPORTE AUTOLOG — collect_cil_data AUTO-FOLLOW")
    lines.append("=" * 62)
    lines.append(f"  Frames auto-follow : {n}")
    lines.append(f"  Duración sim       : {duration_sim_s:.1f} s")
    lines.append(f"  Duración real      : {duration_wall:.1f} s")
    lines.append("")
    lines.append("── OSCILACIÓN ──────────────────────────────────────────")
    lines.append(f"  Cruces de cero (error_norm) : {error_crossings}")
    lines.append(f"  Cruces de cero (steer)      : {steer_crossings}")
    lines.append(f"  Frecuencia (cruces/s)       : {rate_crossings:.2f}  {'⚠ ALTO' if rate_crossings > 1.5 else '✓ OK'}")
    lines.append(f"  Valor esperado estable      : < 1.0 cruces/s")
    lines.append("")
    lines.append("── ERROR DE CARRIL ─────────────────────────────────────")
    lines.append(f"  Error medio (error_norm)    : {mean_error:+.4f}  (0=centro, +1=derecha)")
    lines.append(f"  RMS error                   : {rms_error:.4f}  {'⚠ ALTO' if rms_error > 0.25 else '✓ OK'}")
    lines.append(f"  Steering max (|steer|)      : {max_steer:.4f} rad")
    lines.append(f"  Confianza media             : {mean_conf:.3f}  {'⚠ BAJA' if mean_conf < 0.5 else '✓ OK'}")
    lines.append(f"  LANE_OFFSET esperado        : +{0.12:.2f} (carril derecho)")
    lines.append("")
    lines.append("── DETECCIÓN DE SEMILLAS ────────────────────────────────")
    lines.append(f"  Pérdida de semilla          : {seed_lost_pct:.1f}%  {'⚠ ALTO' if seed_lost_pct > 10 else '✓ OK'}")
    lines.append("")
    lines.append("── ROAD FRACTION (avg_road_frac) ────────────────────────")
    lines.append(f"  Media                       : {mean_frac:.3f}")
    lines.append(f"  Umbral intersección         : 0.60")
    lines.append(f"  % frames con road_frac>0.6  : {inter_detected_pct:.1f}%")
    lines.append("")
    lines.append("── INTERSECCIONES (GPS vs algo) ─────────────────────────")
    lines.append(f"  Mundo: {len(WORLD_INTERSECTIONS)} intersecciones reales")
    lines.append(f"    {WORLD_INTERSECTIONS[0]['label']} ({WORLD_INTERSECTIONS[0]['x']:.0f},{WORLD_INTERSECTIONS[0]['y']:.0f})")
    lines.append(f"    {WORLD_INTERSECTIONS[1]['label']} ({WORLD_INTERSECTIONS[1]['x']:.0f},{WORLD_INTERSECTIONS[1]['y']:.0f})")
    lines.append(f"    {WORLD_INTERSECTIONS[2]['label']} ({WORLD_INTERSECTIONS[2]['x']:.0f},{WORLD_INTERSECTIONS[2]['y']:.0f})")
    lines.append(f"  Radio zona cruce            : {INTER_RADIUS_WBT} m")
    lines.append(f"  Verdaderos positivos (TP)   : {inter_tp}")
    lines.append(f"  Falsos positivos (FP)       : {inter_fp}  {'⚠ HAY FP' if inter_fp > 0 else '✓ OK'}")
    lines.append(f"  Falsos negativos (FN)       : {inter_fn}  {'⚠ HAY FN' if inter_fn > 0 else '✓ OK'}")
    lines.append(f"  Frames GPS='unknown'        : {road_unknown}  (fuera de segmentos conocidos)")
    lines.append("")
    lines.append("── RATE LIMITER ─────────────────────────────────────────")
    lines.append(f"  RATE_LIMIT actual           : 0.05 rad/ciclo")
    lines.append(f"  % frames saturados          : {saturated_pct:.1f}%  {'⚠ SATURADO' if saturated_pct > 30 else '✓ OK'}")
    lines.append("")
    lines.append("── DIAGNÓSTICO ──────────────────────────────────────────")

    # Lógica de diagnóstico
    issues = []
    if rate_crossings > 1.5:
        issues.append("  ⚠  OSCILACIÓN: más de 1.5 cruces/s → reducir KP o SMOOTH_ALPHA")
    if rms_error > 0.3:
        issues.append("  ⚠  ERROR ALTO: RMS>0.3 → el coche se desvía mucho del carril")
    if mean_error < -0.05:
        issues.append("  ⚠  SESGO IZQUIERDO: error medio negativo → revisar LANE_OFFSET")
    if seed_lost_pct > 10:
        issues.append("  ⚠  SEED LOSS: asfalto no detectado >10% → ajustar ROAD_THRESH")
    if mean_conf < 0.5:
        issues.append("  ⚠  CONFIANZA BAJA: pocos píxeles de carretera en ROI → bajar ROI_TOP")
    if inter_fp > inter_tp and inter_tp + inter_fp > 5:
        issues.append("  ⚠  INTERSECCIÓN: más FP que TP → umbral 0.60 demasiado bajo")
    if saturated_pct > 50:
        issues.append("  ⚠  RATE LIMITER: saturado >50% → el control no sigue al error; subir RATE_LIMIT")
    if not issues:
        issues.append("  ✓  Sin problemas detectados — comportamiento esperado")
    lines.extend(issues)
    lines.append("=" * 62)

    report = "\n".join(lines)
    print(report)

    report_path = os.path.join(os.path.dirname(log_path), "autolog_report.txt")
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\n[Reporte guardado en {report_path}]")

    # ── Plot opcional ─────────────────────────────────────────────────────────
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            fig, axes = plt.subplots(3, 2, figsize=(14, 10))
            fig.suptitle("Auto-follow log analysis — collect_cil_data", fontsize=12)

            t = [r["sim_ms"] / 1000.0 for r in rows]

            # 1. error_norm y smooth_center
            ax = axes[0, 0]
            ax.plot(t, error_norms,    alpha=0.5, lw=0.8, label="error_norm")
            ax.plot(t, smooth_centers, lw=1.2,    label="smooth_center")
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_title("error_norm vs smooth_center")
            ax.set_ylabel("normalizado [-1,1]"); ax.legend(fontsize=7)

            # 2. steering
            ax = axes[0, 1]
            ax.plot(t, steers, lw=1.0, color="orange")
            ax.axhline(0, color="gray", lw=0.5)
            ax.axhline( 0.5, color="red", lw=0.5, ls="--", label="MAX_ANGLE")
            ax.axhline(-0.5, color="red", lw=0.5, ls="--")
            ax.set_title("Steering angle (rad)")
            ax.set_ylabel("rad"); ax.legend(fontsize=7)

            # 3. confidence
            ax = axes[1, 0]
            ax.plot(t, confidences, lw=0.8, color="green")
            ax.axhline(0.15, color="red", lw=0.8, ls="--", label="umbral 0.15")
            ax.set_title("Confidence flood fill")
            ax.set_ylabel("[0,1]"); ax.legend(fontsize=7)

            # 4. avg_road_frac
            ax = axes[1, 1]
            ax.plot(t, road_fracs, lw=0.8, color="purple")
            ax.axhline(0.60, color="red", lw=0.8, ls="--", label="umbral intersección")
            ax.fill_between(t, 0, [0.6 if v > 0.6 else 0 for v in road_fracs],
                            alpha=0.2, color="red")
            ax.set_title("avg_road_frac (>0.6 = intersección)")
            ax.set_ylabel("fracción"); ax.legend(fontsize=7)

            # 5. Trayectoria GPS + intersecciones del mundo
            ax = axes[2, 0]
            valid_gps = [(r["gps_x"], r["gps_y"]) for r in rows
                         if not math.isnan(r["gps_x"])]
            if valid_gps:
                xs, ys = zip(*valid_gps)
                sc = ax.scatter(xs, ys, c=range(len(xs)), cmap="viridis",
                                s=2, linewidths=0)
                plt.colorbar(sc, ax=ax, label="frame")
            for inter in WORLD_INTERSECTIONS:
                circ = patches.Circle((inter["x"], inter["y"]),
                                      INTER_RADIUS_WBT, color="red",
                                      fill=False, lw=1.5, ls="--")
                ax.add_patch(circ)
                ax.text(inter["x"], inter["y"] + INTER_RADIUS_WBT + 2,
                        inter["label"], fontsize=6, ha="center", color="red")
            ax.set_aspect("equal"); ax.set_title("Trayectoria GPS + intersecciones mundo")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

            # 6. n_valid_rows
            ax = axes[2, 1]
            ax.plot(t, n_valid, lw=0.7, color="teal")
            ax.set_title("n_valid_rows (filas con asfalto en ROI)")
            ax.set_ylabel("filas")
            ax.set_xlabel("tiempo (s)")

            plt.tight_layout()
            plot_path = os.path.join(os.path.dirname(log_path), "autolog_plots.png")
            plt.savefig(plot_path, dpi=120)
            print(f"[Plot guardado en {plot_path}]")
        except ImportError:
            print("[WARN] matplotlib no disponible — omitiendo plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",  default="data/autolog.csv",
                        help="Ruta al autolog (default: data/autolog.csv)")
    parser.add_argument("--plot", action="store_true",
                        help="Generar autolog_plots.png")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    log_path = str(base / args.log)
    analyze(log_path, plot=args.plot)
