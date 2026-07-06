import cv2
import numpy as np
import json
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from enum import Enum


SEUIL_ANGLE_HORIZONTAL: float = 20.0
SEUIL_ANGLE_VERTICAL:   float = 70.0
SEUIL_LARGEUR_SUPERFICIELLE: float = 6.0
SEUIL_LARGEUR_PROFONDE:      float = 12.0
SEUIL_TRAVERSEE: float = 0.65
DEFAULT_PX_TO_MM = 0.5


class Orientation(str, Enum):
    HORIZONTALE = "horizontale"
    VERTICALE   = "verticale"
    INCLINEE    = "inclinée"
    INCONNUE    = "inconnue"


class Localisation(str, Enum):
    SUPERFICIELLE = "superficielle"
    PROFONDE      = "profonde"
    TRANSVERSALE  = "transversale"
    INCONNUE      = "inconnue"


class Severity(str, Enum):
    HAIRLINE  = "hairline"
    FINE      = "fine"
    MEDIUM    = "medium"
    WIDE      = "wide"
    VERY_WIDE = "very_wide"
    UNKNOWN   = "unknown"


@dataclass
class CrackMeasurements:
    crack_id:       int
    area_px:        float = 0.0
    length_px:      float = 0.0
    width_mean_px:  float = 0.0
    width_max_px:   float = 0.0
    area_mm2:       Optional[float] = None
    length_mm:      Optional[float] = None
    width_mean_mm:  Optional[float] = None
    width_max_mm:   Optional[float] = None
    angle_deg:      float = 0.0
    orientation:    Orientation = Orientation.INCONNUE
    localisation:   Localisation = Localisation.INCONNUE
    sinuosity:      float = 1.0
    aspect_ratio:   float = 1.0
    extent:         float = 0.0
    solidity:       float = 0.0
    bbox:           Tuple[int,int,int,int] = (0,0,0,0)
    bbox_ratio_w:   float = 0.0
    bbox_ratio_h:   float = 0.0
    crack_coverage: float = 0.0
    severity:       Severity = Severity.UNKNOWN
    danger_index:   float = 0.0
    confidence:     float = 1.0


@dataclass
class FrameAnalysis:
    image_path:       str = ""
    image_width:      int = 0
    image_height:     int = 0
    n_cracks:         int = 0
    total_area_px:    float = 0.0
    crack_density:    float = 0.0
    max_severity:     Severity = Severity.UNKNOWN
    max_localisation: Localisation = Localisation.INCONNUE
    max_danger:       float = 0.0
    cracks:           List[CrackMeasurements] = field(default_factory=list)
    px_to_mm:         float = DEFAULT_PX_TO_MM


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    skel   = np.zeros_like(mask)
    img    = mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        sub    = cv2.subtract(img, opened)
        skel   = cv2.bitwise_or(skel, sub)
        img    = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _compute_orientation(mask: np.ndarray) -> Tuple[float, Orientation]:
    rows, cols = np.where(mask > 0)
    if len(rows) < 5:
        return 0.0, Orientation.INCONNUE

    pts = np.column_stack([cols, rows]).astype(np.float32)
    _, eigvecs = cv2.PCACompute(pts, mean=None)
    angle_rad = math.atan2(float(eigvecs[0, 1]), float(eigvecs[0, 0]))
    angle_deg = math.degrees(angle_rad)

    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    abs_a = abs(angle_deg)
    if abs_a < SEUIL_ANGLE_HORIZONTAL:
        label = Orientation.HORIZONTALE
    elif abs_a > SEUIL_ANGLE_VERTICAL:
        label = Orientation.VERTICALE
    else:
        label = Orientation.INCLINEE

    return round(angle_deg, 2), label


def _compute_localisation(
    mask: np.ndarray,
    width_median_px: float,
    bbox_w: int,
    bbox_h: int,
    image_w: int,
    image_h: int,
) -> Tuple[Localisation, float, float]:
    ratio_w = bbox_w / image_w if image_w > 0 else 0.0
    ratio_h = bbox_h / image_h if image_h > 0 else 0.0

    if ratio_w >= SEUIL_TRAVERSEE or ratio_h >= SEUIL_TRAVERSEE:
        return Localisation.TRANSVERSALE, ratio_w, ratio_h
    if width_median_px > SEUIL_LARGEUR_PROFONDE:
        return Localisation.PROFONDE, ratio_w, ratio_h
    if width_median_px < SEUIL_LARGEUR_SUPERFICIELLE:
        return Localisation.SUPERFICIELLE, ratio_w, ratio_h
    return Localisation.PROFONDE, ratio_w, ratio_h


def _compute_width_profile(mask: np.ndarray) -> Tuple[float, float, float]:
    try:
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(mask > 0)
    except ImportError:
        dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

    nonzero_dist = dist[mask > 0]
    if len(nonzero_dist) == 0:
        return 0.0, 0.0, 0.0

    w_median = float(np.median(nonzero_dist)) * 2
    w_mean   = float(np.mean(nonzero_dist))   * 2
    w_max    = float(np.max(nonzero_dist))    * 2
    return w_median, w_mean, w_max


def _find_skeleton_endpoints(skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    skel_pts = np.column_stack(np.where(skeleton > 0))
    if len(skel_pts) < 2:
        return skel_pts[0], skel_pts[0]

    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel)
    endpoints_mask = (skeleton > 0) & (neighbor_count == 2)
    endpoint_pts   = np.column_stack(np.where(endpoints_mask))

    candidates = endpoint_pts if len(endpoint_pts) >= 2 else skel_pts
    dists = np.linalg.norm(candidates[:, None].astype(float) - candidates[None, :].astype(float), axis=2)
    i, j  = np.unravel_index(np.argmax(dists), dists.shape)
    return candidates[i], candidates[j]


def _compute_sinuosity(skeleton: np.ndarray) -> float:
    pts = np.column_stack(np.where(skeleton > 0))
    if len(pts) < 2:
        return 1.0
    skel_length = float(len(pts))
    ep1, ep2    = _find_skeleton_endpoints(skeleton)
    end_to_end  = float(np.linalg.norm(ep1.astype(float) - ep2.astype(float)))
    if end_to_end < 1e-6:
        return 1.0
    return round(min(skel_length / end_to_end, 10.0), 3)


def _severity_from_width(width_mm: float) -> Severity:
    if width_mm < 0.1:  return Severity.HAIRLINE
    if width_mm < 1.0:  return Severity.FINE
    if width_mm < 5.0:  return Severity.MEDIUM
    if width_mm < 15.0: return Severity.WIDE
    return Severity.VERY_WIDE


def _compute_danger_index(
    localisation:   Localisation,
    orientation:    Orientation,
    width_mean_px:  float,
    crack_coverage: float,
) -> float:
    scores_loc = {
        Localisation.TRANSVERSALE:  1.00,
        Localisation.PROFONDE:      0.70,
        Localisation.SUPERFICIELLE: 0.25,
        Localisation.INCONNUE:      0.40,
    }
    scores_ori = {
        Orientation.VERTICALE:   0.80,
        Orientation.INCLINEE:    0.60,
        Orientation.HORIZONTALE: 0.40,
        Orientation.INCONNUE:    0.40,
    }
    score_loc = scores_loc.get(localisation, 0.40)
    score_ori = scores_ori.get(orientation, 0.40)
    score_lar = min(width_mean_px / 20.0, 1.0)
    score_cov = float(np.clip(crack_coverage, 0.0, 1.0))

    danger = (
        0.45 * score_loc
        + 0.20 * score_ori
        + 0.20 * score_lar
        + 0.15 * score_cov
    )
    return round(float(np.clip(danger, 0.0, 1.0)), 4)


def analyze_instance(
    mask_binary: np.ndarray,
    crack_id:    int,
    image_w:     int,
    image_h:     int,
    confidence:  float = 1.0,
    px_to_mm:    float = DEFAULT_PX_TO_MM,
) -> CrackMeasurements:
    m    = CrackMeasurements(crack_id=crack_id, confidence=confidence)
    mask = (mask_binary > 127).astype(np.uint8) * 255
    area_px = float(np.count_nonzero(mask))

    if area_px == 0:
        return m
    m.area_px = area_px
    m.crack_coverage = area_px / (image_w * image_h) if (image_w * image_h) > 0 else 0.0

    x, y, bw, bh = cv2.boundingRect(mask)
    m.bbox   = (x, y, bw, bh)
    m.extent = area_px / (bw * bh) if (bw * bh) > 0 else 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull      = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        m.solidity = area_px / hull_area if hull_area > 0 else 1.0
    else:
        m.solidity = 1.0

    skeleton   = _skeletonize(mask)
    skel_len   = float(np.count_nonzero(skeleton))
    m.length_px = skel_len if skel_len > 0 else max(bw, bh)
    m.sinuosity = _compute_sinuosity(skeleton)

    w_median, w_mean, w_max = _compute_width_profile(mask)
    m.width_mean_px = round(w_mean, 2)
    m.width_max_px  = round(w_max,  2)
    m.aspect_ratio  = round(m.length_px / w_mean, 2) if w_mean > 0 else 0.0

    m.angle_deg, m.orientation = _compute_orientation(mask)

    m.localisation, m.bbox_ratio_w, m.bbox_ratio_h = _compute_localisation(
        mask, w_median, bw, bh, image_w, image_h
    )

    m.area_mm2      = round(area_px    * (px_to_mm ** 2), 3)
    m.length_mm     = round(m.length_px  * px_to_mm, 3)
    m.width_mean_mm = round(w_mean       * px_to_mm, 3)
    m.width_max_mm  = round(w_max        * px_to_mm, 3)

    m.severity     = _severity_from_width(m.width_mean_mm)
    m.danger_index = _compute_danger_index(
        m.localisation, m.orientation, w_mean, m.crack_coverage
    )
    return m


def analyze_frame(
    image:      np.ndarray,
    masks:      List[np.ndarray],
    scores:     List[float],
    image_path: str = "",
    px_to_mm:   float = DEFAULT_PX_TO_MM,
) -> FrameAnalysis:
    h, w   = image.shape[:2]
    result = FrameAnalysis(image_path=image_path, image_width=w, image_height=h, px_to_mm=px_to_mm)

    if not masks:
        return result

    result.n_cracks = len(masks)
    total_area      = 0.0
    severities      = []
    localisations   = []
    dangers         = []

    for i, (mask, score) in enumerate(zip(masks, scores)):
        meas = analyze_instance(mask, crack_id=i, image_w=w, image_h=h,
                                confidence=float(score), px_to_mm=px_to_mm)
        result.cracks.append(meas)
        total_area += meas.area_px
        severities.append(meas.severity)
        localisations.append(meas.localisation)
        dangers.append(meas.danger_index)

    result.total_area_px = total_area
    result.crack_density = round(total_area / (w * h) * 100, 3)
    result.max_danger    = round(max(dangers), 4) if dangers else 0.0

    sev_order = [Severity.HAIRLINE, Severity.FINE, Severity.MEDIUM, Severity.WIDE, Severity.VERY_WIDE]
    loc_order = [Localisation.SUPERFICIELLE, Localisation.PROFONDE, Localisation.TRANSVERSALE]
    if severities:
        result.max_severity = max(
            [s for s in severities if s != Severity.UNKNOWN],
            key=lambda s: sev_order.index(s) if s in sev_order else -1,
            default=Severity.UNKNOWN,
        )
    if localisations:
        result.max_localisation = max(
            [l for l in localisations if l != Localisation.INCONNUE],
            key=lambda l: loc_order.index(l) if l in loc_order else -1,
            default=Localisation.INCONNUE,
        )
    return result


LOCALISATION_COLORS = {
    Localisation.SUPERFICIELLE: (0,   200, 100),
    Localisation.PROFONDE:      (0,   140, 255),
    Localisation.TRANSVERSALE:  (0,   0,   220),
    Localisation.INCONNUE:      (128, 128, 128),
}


def draw_analysis(
    image:          np.ndarray,
    frame_analysis: FrameAnalysis,
    masks:          List[np.ndarray],
    alpha:          float = 0.4,
    draw_skeleton:  bool  = True,
) -> np.ndarray:
    vis     = image.copy()
    overlay = image.copy()
    font    = cv2.FONT_HERSHEY_SIMPLEX

    for meas, mask in zip(frame_analysis.cracks, masks):
        color    = LOCALISATION_COLORS[meas.localisation]
        bin_mask = (mask > 127).astype(np.uint8)

        overlay[bin_mask == 1] = color
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
        overlay = vis.copy()

        if draw_skeleton:
            skel = _skeletonize((bin_mask * 255).astype(np.uint8))
            vis[skel > 0] = (255, 255, 255)

        bx, by, bw_b, bh_b = meas.bbox
        cv2.rectangle(vis, (bx, by), (bx + bw_b, by + bh_b), color, 2)

        label_lines = [
            f"#{meas.crack_id}  {meas.localisation.value}  conf={meas.confidence:.2f}",
            f"L={meas.length_mm:.1f}mm  W={meas.width_mean_mm:.1f}mm",
            f"{meas.orientation.value} ({meas.angle_deg:.1f}°)  sin={meas.sinuosity:.2f}",
            f"danger={meas.danger_index:.2f}  {meas.severity.value}",
        ]
        ty = max(by - 5, 15)
        for j, line in enumerate(reversed(label_lines)):
            cv2.putText(vis, line, (bx, ty - j * 16), font, 0.42, (0,0,0), 3)
            cv2.putText(vis, line, (bx, ty - j * 16), font, 0.42, (255,255,255), 1)

    summary = [
        f"Fissures : {frame_analysis.n_cracks}",
        f"Densite  : {frame_analysis.crack_density:.2f}%",
        f"Danger   : {frame_analysis.max_danger:.2f}",
        f"Local.   : {frame_analysis.max_localisation.value}",
    ]
    for j, line in enumerate(summary):
        cv2.putText(vis, line, (10, 20 + j * 22), font, 0.6, (0,0,0), 3)
        cv2.putText(vis, line, (10, 20 + j * 22), font, 0.6, (255,255,255), 1)

    return vis


def analysis_to_dict(frame: FrameAnalysis) -> dict:
    d = asdict(frame)
    d["max_severity"]     = frame.max_severity.value
    d["max_localisation"] = frame.max_localisation.value
    for crack in d["cracks"]:
        crack["severity"]     = crack["severity"]     if isinstance(crack["severity"],     str) else crack["severity"].value
        crack["localisation"] = crack["localisation"] if isinstance(crack["localisation"], str) else crack["localisation"].value
        crack["orientation"]  = crack["orientation"]  if isinstance(crack["orientation"],  str) else crack["orientation"].value
    return d


def save_analysis_report(frame: FrameAnalysis, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_to_dict(frame), f, indent=2, ensure_ascii=False)
    print(f"  Rapport JSON : {output_path}")


def print_analysis_summary(frame: FrameAnalysis):
    print("\n" + "─"*60)
    print(f"  Image          : {frame.image_path}")
    print(f"  Résolution     : {frame.image_width}×{frame.image_height} px")
    print(f"  Fissures       : {frame.n_cracks}")
    print(f"  Densité        : {frame.crack_density:.3f}%")
    print(f"  Sévérité max   : {frame.max_severity.value}")
    print(f"  Localisation   : {frame.max_localisation.value}")
    print(f"  Indice danger  : {frame.max_danger:.4f}")
    print("─"*60)
    for m in frame.cracks:
        print(f"\n  Fissure #{m.crack_id}  [confiance={m.confidence:.2f}]")
        print(f"    Orientation  : {m.orientation.value} ({m.angle_deg:.1f}°)")
        print(f"    Localisation : {m.localisation.value}")
        print(f"    Sévérité     : {m.severity.value}")
        print(f"    Indice danger: {m.danger_index:.4f}")
        print(f"    Longueur     : {m.length_mm:.2f} mm  ({m.length_px:.0f} px)")
        print(f"    Largeur moy  : {m.width_mean_mm:.2f} mm  ({m.width_mean_px:.2f} px)")
        print(f"    Largeur max  : {m.width_max_mm:.2f} mm  ({m.width_max_px:.2f} px)")
        print(f"    Surface      : {m.area_mm2:.2f} mm²  ({m.area_px:.0f} px²)")
        print(f"    Sinuosité    : {m.sinuosity:.3f}")
        print(f"    Rapport L/l  : {m.aspect_ratio:.1f}")
        print(f"    Couverture   : {m.crack_coverage*100:.3f}%")
        print(f"    BBox ratio   : w={m.bbox_ratio_w:.2f} h={m.bbox_ratio_h:.2f}")
    print("─"*60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask",     type=str, default=None)
    parser.add_argument("--image",    type=str, default=None)
    parser.add_argument("--px-to-mm", type=float, default=DEFAULT_PX_TO_MM)
    args = parser.parse_args()

    if args.mask:
        mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print(f"❌ Impossible de lire : {args.mask}")
        else:
            base_img = cv2.imread(args.image) if args.image else np.zeros((*mask_img.shape, 3), dtype=np.uint8)
            h, w = base_img.shape[:2]
            frame = analyze_frame(base_img, [mask_img], [1.0], args.mask, args.px_to_mm)
            print_analysis_summary(frame)
    else:
        H, W = 200, 400
        test_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.line(test_mask, (20, 180), (380, 20), 255, 8)
        test_img  = np.zeros((H, W, 3), dtype=np.uint8)
        frame     = analyze_frame(test_img, [test_mask], [0.92], "synthetic_test", 0.5)
        print_analysis_summary(frame)
        print("✓ Test autonome réussi.")
