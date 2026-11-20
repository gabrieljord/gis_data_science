"""
Predicted-path overlay that follows the arc of the road.
Real-time LineString pixel route mapping in a separate window with real forward translation and pose accumulation.
Keys:  q = quit, c = toggle tracked corners, b = toggle BEV, s = save route GeoJSON, r = reset route
"""

import json
import math
from pathlib import Path
import queue
import threading
import PIL import Image
from PIL.ExifTags import TAGS

import cv2
import numpy as np

try:
    from shapely.geometry import LineString, Point, mapping
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

DEFAULT_VIDEO = Path(__file__).resolve().parent.parent / "data" / "test.mp4"
VIDEO_PATH = str(DEFAULT_VIDEO if DEFAULT_VIDEO.exists() else Path(r"/home/blujay/code/gis_data_science/tutorial_4/data/test.mp4"))

W, H = 1920, 1080
WINDOW_SIZE = (1280, 720)
MAP_WINDOW_SIZE = (720, 720)


HFOV_DEG = 70.0                       # horizontal field of view of the camera
FOCAL = (W / 2) / np.tan(np.radians(HFOV_DEG / 2))   # focal length in pixels
CAM_HEIGHT = 1.2                      
VP = (0.33 * W, 0.54 * H)             # vanishing point of a straight road (u, v)
DASH_Y = 0.635 * H                    
Z_NEAR = FOCAL * CAM_HEIGHT / (DASH_Y - VP[1])
Z_FAR = 70.0                          # farthest ground distance used (m)
Z_TRACK_FAR = 60.0                    
TRACK_HALF_WIDTH = 6.0                

PATH_WIDTH_M = 3.0                    
PATH_OFFSET_X = 1.1                
PATH_COLOR = (255, 200, 0)
EDGE_COLOR = (255, 255, 255) 
ALPHA_NEAR, ALPHA_FAR = 0.55, 0.10   
KAPPA_MAX = 0.02                     

YAW_EMA = 0.08                        
MIN_TRACKED = 60                      
MIN_USED = 15                         # need this many good tracks to update

BEV_HALF_WIDTH = 10.0                 
BEV_PPM = 6                          
BEV_W = int(2 * BEV_HALF_WIDTH * BEV_PPM)
BEV_H = int((Z_FAR - Z_NEAR) * BEV_PPM)


def ground_to_image(X, Z):
    X, Z = np.asarray(X, np.float64), np.asarray(Z, np.float64)
    return VP[0] + FOCAL * X / Z, VP[1] + FOCAL * CAM_HEIGHT / Z


def build_homographies():
    xw = BEV_HALF_WIDTH
    gx = np.array([-xw, xw, xw, -xw], np.float32)
    gz = np.array([Z_FAR, Z_FAR, Z_NEAR, Z_NEAR], np.float32)
    u, v = ground_to_image(gx, gz)
    img_pts = np.stack([u, v], 1).astype(np.float32)
    ground_pts = np.stack([gx, gz], 1)
    bev_pts = np.array([[0, 0], [BEV_W, 0], [BEV_W, BEV_H], [0, BEV_H]], np.float32)

    i2g = cv2.getPerspectiveTransform(img_pts, ground_pts)   # pixels -> metres
    g2i = np.linalg.inv(i2g)                                  # metres -> pixels
    i2b = cv2.getPerspectiveTransform(img_pts, bev_pts)       # pixels -> BEV pixels
    return i2g, g2i, i2b


I2G, G2I, I2B = build_homographies()


def transform(pts, M):
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2)


def ground_to_bev(X, Z):
    return np.stack([(np.asarray(X) + BEV_HALF_WIDTH) * BEV_PPM,
                     (Z_FAR - np.asarray(Z)) * BEV_PPM], 1)


def build_road_mask():
    """Shi-Tomasi"""
    xw = TRACK_HALF_WIDTH
    gx = np.array([-xw, xw, xw, -xw], np.float32)
    gz = np.array([Z_TRACK_FAR, Z_TRACK_FAR, Z_NEAR, Z_NEAR], np.float32)
    poly = transform(np.stack([gx, gz], 1), G2I)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
    return mask


class EgoMotion:
    def __init__(self, fps):
        self.fps = fps
        self.mask = build_road_mask()
        rows = np.nonzero(self.mask.any(axis=1))[0]
        self.r0, self.r1 = max(int(rows.min()) - 40, 0), min(int(rows.max()) + 40, H)  # road strip
        self.prev_crop = None
        self.pts = np.empty((0, 1, 2), np.float32)
        self.yaw = 0.0                                   # rad per frame (+ = turning right)
        self.step = 0.0                                  # forward displacement this frame (m)
        self.x = 0.0                                     # accumulated metric pose X (m)
        self.y = 0.0                                     # accumulated metric pose Y (m)
        self.heading = 0.0                               # accumulated heading angle (rad)
        self.trajectory = []                             # buffer of (x, y, heading, dz, psi)
        self.n_used = 0

    @property
    def speed_mps(self):
        """Current speed in m/s."""
        return self.step * self.fps

    @property
    def speed_kmh(self):
        """Current speed in km/h."""
        return self.speed_mps * 3.6

    @property
    def kappa(self):
        """Path curvature in 1/m (+ = curving right)."""
        if self.step < 1e-4:
            return 0.0
        return float(np.clip(self.yaw / self.step, -KAPPA_MAX, KAPPA_MAX))

    def _detect(self, crop):
        mask = self.mask[self.r0:self.r1].copy()
        for x, y in self.pts.reshape(-1, 2):
            cv2.circle(mask, (int(x), int(y) - self.r0), 8, 0, -1)
        new = cv2.goodFeaturesToTrack(crop, maxCorners=250, qualityLevel=0.005,
                                      minDistance=8, blockSize=5, mask=mask)
        if new is not None:
            new = new.astype(np.float32) + np.array([0, self.r0], np.float32)
            self.pts = np.concatenate([self.pts, new])

    def update(self, gray):
        crop = gray[self.r0:self.r1]
        off = np.array([0, self.r0], np.float32)
        if self.prev_crop is not None and len(self.pts) >= 10:
            lk = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            p0 = self.pts - off
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_crop, crop, p0, None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(crop, self.prev_crop, nxt, None, **lk)
            err = np.linalg.norm(p0 - back, axis=2).ravel()          # forward-backward check
            good = (st.ravel() == 1) & (st2.ravel() == 1) & (err < 0.5)
            self.pts = (nxt[good] + off).reshape(-1, 1, 2)
            self._estimate((p0[good] + off).reshape(-1, 2), self.pts.reshape(-1, 2))
        else:
            self.step = 0.0

        if len(self.pts) < MIN_TRACKED:
            self._detect(crop)
        self.prev_crop = crop

    def _estimate(self, p0, p1):
        g0, g1 = transform(p0, I2G), transform(p1, I2G)
        z0 = g0[:, 1]
        ok = (z0 > Z_NEAR * 0.9) & (z0 < Z_TRACK_FAR * 1.1) & (g1[:, 1] > 1)
        self.n_used = int(ok.sum())
        if self.n_used < MIN_USED:
            self.step = 0.0
            return
        g0, g1 = g0[ok], g1[ok]

        dz = float(np.clip(np.median(g0[:, 1] - g1[:, 1]), 0.0, 3.0))
        psi = float(np.median(-(g1[:, 0] - g0[:, 0]) / g0[:, 1]))

        self.yaw += YAW_EMA * (psi - self.yaw)
        self.step = dz

        self.heading += psi
        self.x += dz * np.sin(self.heading)
        self.y += dz * np.cos(self.heading)
        self.trajectory.append((self.x, self.y, self.heading, dz, psi))


class PixelRouteMap:
    """
    Tracks vehicle odometry from dynamic ego-motion speed & yaw,
    generating a real-time pixel map representing the driven route
    formatted as an OGC / GeoJSON LineString feature.
    """
    def __init__(self, width=720, height=720, margin=75):
        self.w = width
        self.h = height
        self.margin = margin
        self.reset()

    def reset(self):
        """Reset trajectory and kinematic state."""
        self.x = 0.0               # +X = East (right) in metres
        self.y = 0.0               # +Y = North (forward) in metres
        self.heading = 0.0         # 0 = North (+Y), + = clockwise (East)
        self.total_dist = 0.0      # total distance in metres
        self.current_speed_mps = 0.0
        self.is_moving = False

        # Trajectory storage
        self.metric_coords = [(0.0, 0.0)]
        self.pixel_coords = []

        # Smooth camera viewport parameters
        self.view_cx = 0.0
        self.view_cy = 0.0
        self.view_scale = 3.0       # pixels per metre
        self.initialized_view = False

    def update(self, *args, **kwargs):
        """
        Update vehicle position and heading from EgoMotion state or arguments.
        """
        if len(args) == 1 and isinstance(args[0], EgoMotion):
            ego = args[0]
            self.x = ego.x
            self.y = ego.y
            self.heading = ego.heading
            self.current_speed_mps = ego.speed_mps
            self.is_moving = (ego.step > 0.005)
            self.total_dist = float(sum(p[3] for p in ego.trajectory)) if ego.trajectory else 0.0
            if ego.trajectory:
                self.metric_coords = [(p[0], p[1]) for p in ego.trajectory]
        else:
            yaw_rate = args[0] if len(args) > 0 else kwargs.get("yaw_rate", 0.0)
            step_dist = args[1] if len(args) > 1 else kwargs.get("step_dist", 0.0)
            speed_mps = args[2] if len(args) > 2 else kwargs.get("speed_mps", step_dist * 30.0)
            self.current_speed_mps = speed_mps
            self.is_moving = (step_dist > 0.005)
            if self.is_moving:
                mid_heading = self.heading + 0.5 * yaw_rate
                self.heading = (self.heading + yaw_rate) % (2.0 * math.pi)
                self.x += step_dist * math.sin(mid_heading)
                self.y += step_dist * math.cos(mid_heading)
                self.total_dist += step_dist
                self.metric_coords.append((self.x, self.y))

    def get_feature(self, coordinate_type="pixel"):
        """
        Returns the route as an OGC / GeoJSON standard LineString Feature dictionary.
        coordinate_type: 'pixel' for canvas pixel coords [[u, v], ...],
                         'metric' for local coordinates [[x, y], ...].
        """
        if coordinate_type == "pixel" and self.pixel_coords:
            coords = [[int(round(p[0])), int(round(p[1]))] for p in self.pixel_coords]
        else:
            coords = [[round(p[0], 3), round(p[1], 3)] for p in self.metric_coords]

        heading_deg = math.degrees(self.heading) % 360.0

        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "name": "Driven Route",
                "feature_type": "LineString",
                "coordinate_system": coordinate_type,
                "total_distance_m": round(self.total_dist, 2),
                "current_speed_kmh": round(self.current_speed_mps * 3.6, 1),
                "is_moving": self.is_moving,
                "num_vertices": len(coords),
                "current_heading_deg": round(heading_deg, 1),
                "current_position": coords[-1] if coords else [0, 0],
            }
        }

    def to_shapely(self, coordinate_type="pixel"):
        """Returns the route as a shapely.geometry.LineString."""
        if not HAS_SHAPELY:
            return None
        coords = self.pixel_coords if (coordinate_type == "pixel" and self.pixel_coords) else self.metric_coords
        if len(coords) < 2:
            return None
        return LineString(coords)

    def save_geojson(self, filepath, coordinate_type="pixel"):
        """Saves a GeoJSON FeatureCollection containing the route LineString and endpoints."""
        route_feature = self.get_feature(coordinate_type=coordinate_type)
        coords = route_feature["geometry"]["coordinates"]
        start_pt = coords[0] if coords else [0, 0]
        curr_pt = coords[-1] if coords else [0, 0]

        fc = {
            "type": "FeatureCollection",
            "features": [
                route_feature,
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": start_pt},
                    "properties": {"role": "start_point"}
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": curr_pt},
                    "properties": {
                        "role": "current_position",
                        "speed_kmh": route_feature["properties"]["current_speed_kmh"],
                        "heading_deg": route_feature["properties"]["current_heading_deg"],
                        "total_distance_m": route_feature["properties"]["total_distance_m"]
                    }
                }
            ]
        }
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(fc, f, indent=2)
        return path

    def render(self):
        """Renders the pixel map showing the LineString route, markers, dynamic speed and HUD."""
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        canvas[:] = (24, 26, 28)  # sleek dark slate theme

        pts_m = np.array(self.metric_coords, dtype=np.float32)
        min_x, min_y = pts_m.min(axis=0)
        max_x, max_y = pts_m.max(axis=0)

        # Lookahead buffer along current heading
        ahead_x = self.x + 25.0 * math.sin(self.heading)
        ahead_y = self.y + 25.0 * math.cos(self.heading)
        min_x = min(min_x, ahead_x)
        max_x = max(max_x, ahead_x)
        min_y = min(min_y, ahead_y)
        max_y = max(max_y, ahead_y)

        target_cx = float((min_x + max_x) / 2.0)
        target_cy = float((min_y + max_y) / 2.0)
        span_x = max(float(max_x - min_x), 40.0)
        span_y = max(float(max_y - min_y), 40.0)

        avail_w = self.w - 2 * self.margin
        avail_h = self.h - 2 * self.margin
        target_scale = float(min(avail_w / span_x, avail_h / span_y))

        # Viewport exponential smoothing
        if not self.initialized_view:
            self.view_cx = target_cx
            self.view_cy = target_cy
            self.view_scale = target_scale
            self.initialized_view = True
        else:
            self.view_cx += 0.15 * (target_cx - self.view_cx)
            self.view_cy += 0.15 * (target_cy - self.view_cy)
            self.view_scale += 0.15 * (target_scale - self.view_scale)

        cx, cy, s = self.view_cx, self.view_cy, self.view_scale

        # Grid spacing
        target_grid_m = 75.0 / s
        pow10 = 10 ** math.floor(math.log10(max(target_grid_m, 1.0)))
        candidates = [1.0 * pow10, 2.0 * pow10, 5.0 * pow10, 10.0 * pow10]
        grid_m = min(candidates, key=lambda c: abs(c - target_grid_m))

        left_m = cx - (self.w / 2.0) / s
        right_m = cx + (self.w / 2.0) / s
        bottom_m = cy - (self.h / 2.0) / s
        top_m = cy + (self.h / 2.0) / s

        start_grid_x = math.floor(left_m / grid_m) * grid_m
        start_grid_y = math.floor(bottom_m / grid_m) * grid_m

        # Vectorized coordinate transform
        u_arr = np.round(self.w / 2.0 + (pts_m[:, 0] - cx) * s).astype(np.int32)
        v_arr = np.round(self.h / 2.0 - (pts_m[:, 1] - cy) * s).astype(np.int32)
        pts_px = np.column_stack([u_arr, v_arr])
        self.pixel_coords = pts_px.tolist()

        # Draw grid lines
        gx = start_grid_x
        while gx <= right_m:
            gu = int(round(self.w / 2.0 + (gx - cx) * s))
            if 0 <= gu < self.w:
                cv2.line(canvas, (gu, 0), (gu, self.h), (38, 42, 46), 1)
            gx += grid_m

        gy = start_grid_y
        while gy <= top_m:
            gv = int(round(self.h / 2.0 - (gy - cy) * s))
            if 0 <= gv < self.h:
                cv2.line(canvas, (0, gv), (self.w, gv), (38, 42, 46), 1)
            gy += grid_m

        # Draw LineString path
        if len(pts_px) >= 2:
            # Polyline glow / outer line
            cv2.polylines(canvas, [pts_px.reshape(-1, 1, 2)], isClosed=False,
                          color=(140, 90, 0), thickness=5, lineType=cv2.LINE_AA)
            # Main LineString polyline
            cv2.polylines(canvas, [pts_px.reshape(-1, 1, 2)], isClosed=False,
                          color=(255, 210, 30), thickness=2, lineType=cv2.LINE_AA)

            # Periodic vertex breadcrumbs
            step_crumb = max(1, len(pts_px) // 25)
            for idx in range(0, len(pts_px), step_crumb):
                cv2.circle(canvas, tuple(pts_px[idx]), 2, (200, 240, 255), -1, lineType=cv2.LINE_AA)

        # Start Marker
        start_u, start_v = pts_px[0]
        if 0 <= start_u < self.w and 0 <= start_v < self.h:
            cv2.circle(canvas, (start_u, start_v), 7, (40, 200, 80), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (start_u, start_v), 11, (60, 240, 100), 1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, "START", (start_u + 12, start_v + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 240, 120), 1, cv2.LINE_AA)

        # Vehicle Position & Direction Arrow
        curr_u, curr_v = pts_px[-1]
        vehicle_color = (0, 220, 255) if self.is_moving else (0, 120, 255)
        cv2.circle(canvas, (curr_u, curr_v), 6, vehicle_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (curr_u, curr_v), 10, (255, 255, 255) if self.is_moving else (120, 120, 120), 2, lineType=cv2.LINE_AA)

        arrow_len, arrow_base = 18.0, 7.0
        sin_h, cos_h = math.sin(self.heading), math.cos(self.heading)
        tip = (int(round(curr_u + arrow_len * sin_h)),
               int(round(curr_v - arrow_len * cos_h)))
        left_c = (int(round(curr_u - (arrow_len * 0.4) * sin_h - arrow_base * cos_h)),
                  int(round(curr_v + (arrow_len * 0.4) * cos_h - arrow_base * sin_h)))
        right_c = (int(round(curr_u - (arrow_len * 0.4) * sin_h + arrow_base * cos_h)),
                   int(round(curr_v + (arrow_len * 0.4) * cos_h + arrow_base * sin_h)))

        arrow_poly = np.array([tip, left_c, right_c], dtype=np.int32)
        cv2.fillPoly(canvas, [arrow_poly], vehicle_color, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [arrow_poly], True, (0, 60, 140), 1, lineType=cv2.LINE_AA)

        if not self.is_moving:
            cv2.putText(canvas, "STOPPED", (curr_u + 14, curr_v - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255), 1, cv2.LINE_AA)

        # HUD Panel (Top-Left)
        panel_w, panel_h = 320, 150
        overlay = canvas.copy()
        cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (12, 14, 16), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (15, 15), (15 + panel_w, 15 + panel_h), (60, 70, 80), 1)

        heading_deg = math.degrees(self.heading) % 360.0
        speed_kmh = self.current_speed_mps * 3.6
        speed_str = f"{speed_kmh:4.1f} km/h ({self.current_speed_mps:4.1f} m/s)"
        status_str = "MOVING" if self.is_moving else "STOPPED"
        status_color = (80, 240, 120) if self.is_moving else (0, 140, 255)

        cv2.putText(canvas, "PIXEL ROUTE MAP", (25, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Format: LineString Feature", (25, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)
        
        cv2.putText(canvas, f"Speed: {speed_str}", (25, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"[{status_str}]", (245, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)

        cv2.putText(canvas, f"Distance: {self.total_dist:.1f} m", (25, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Heading: {heading_deg:.1f} deg", (25, 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Vertices: {len(pts_px)} pts  |  Scale: {s:.2f} px/m", (25, 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # Compass Rose (Top-Right)
        compass_cx, compass_cy = self.w - 45, 45
        cv2.circle(canvas, (compass_cx, compass_cy), 22, (32, 36, 40), -1)
        cv2.circle(canvas, (compass_cx, compass_cy), 22, (70, 80, 90), 1)
        cv2.arrowedLine(canvas, (compass_cx, compass_cy + 12), (compass_cx, compass_cy - 12),
                        (50, 120, 255), 2, tipLength=0.35)
        cv2.putText(canvas, "N", (compass_cx - 5, compass_cy - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Scale Bar (Bottom-Right)
        bar_len_m = grid_m
        bar_len_px = int(round(bar_len_m * s))
        if bar_len_px > 0:
            bar_x1 = self.w - 30
            bar_x0 = bar_x1 - bar_len_px
            bar_y = self.h - 25
            cv2.line(canvas, (bar_x0, bar_y), (bar_x1, bar_y), (220, 220, 220), 2)
            cv2.line(canvas, (bar_x0, bar_y - 4), (bar_x0, bar_y + 4), (220, 220, 220), 2)
            cv2.line(canvas, (bar_x1, bar_y - 4), (bar_x1, bar_y + 4), (220, 220, 220), 2)
            scale_lbl = f"{int(bar_len_m)} m" if bar_len_m >= 1 else f"{bar_len_m:.1f} m"
            (tw, _), _ = cv2.getTextSize(scale_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(canvas, scale_lbl, (bar_x0 + (bar_len_px - tw) // 2, bar_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

        return canvas


def path_on_ground(kappa, s_max=120.0, n=160):
    s = np.linspace(0, s_max, n)
    if abs(kappa) < 1e-6:
        phi, cx, cz = np.zeros_like(s), np.zeros_like(s), s
    else:
        phi = kappa * s
        cx, cz = (1 - np.cos(phi)) / kappa, np.sin(phi) / kappa
    cx = cx + PATH_OFFSET_X
    nx, nz = np.cos(phi), -np.sin(phi) # unit normal pointing right
    hw = PATH_WIDTH_M / 2
    left = np.stack([cx - hw * nx, cz - hw * nz], 1)
    right = np.stack([cx + hw * nx, cz + hw * nz], 1)
    keep = (cz >= Z_NEAR * 0.98) & (cz <= Z_FAR)
    return left[keep], right[keep]


def draw_path(img, kappa):
    left_g, right_g = path_on_ground(kappa)
    if len(left_g) < 2:
        return
    left, right = transform(left_g, G2I), transform(right_g, G2I)

    both = np.concatenate([left, right])
    x0, y0 = np.floor(both.min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(both.max(axis=0)).astype(int) + 3
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    shift = np.array([x0, y0])
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    n = len(left)
    for i in range(n - 1):
        a = ALPHA_NEAR + (ALPHA_FAR - ALPHA_NEAR) * i / (n - 1)
        quad = np.round(np.array([left[i], left[i + 1], right[i + 1], right[i]]) - shift)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), int(a * 255))
    a = mask[..., None].astype(np.float32) / 255
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - a) + np.array(PATH_COLOR, np.float32) * a).astype(np.uint8)

    for edge in (left, right):
        cv2.polylines(img, [np.round(edge).astype(np.int32)], False, EDGE_COLOR, 2, cv2.LINE_AA)


def draw_bev_inset(img, clean, ego, kappa):
    bev = cv2.warpPerspective(clean, I2B, (BEV_W, BEV_H))
    left, right = path_on_ground(kappa)
    for edge in (left, right):
        if len(edge) >= 2:
            pts = ground_to_bev(edge[:, 0], edge[:, 1])
            cv2.polylines(bev, [np.round(pts).astype(np.int32)], False, PATH_COLOR, 2, cv2.LINE_AA)
    if len(ego.pts):
        for x, y in transform(ego.pts.reshape(-1, 2), I2B):
            if 0 <= x < BEV_W and 0 <= y < BEV_H:
                cv2.circle(bev, (int(x), int(y)), 2, (0, 255, 255), -1)
    x0, y0 = W - BEV_W - 20, 20
    cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + BEV_W + 2, y0 + BEV_H + 2), (255, 255, 255), 2)
    img[y0:y0 + BEV_H, x0:x0 + BEV_W] = bev


def draw_hud(img, ego, kappa):
    radius = "straight" if abs(kappa) < 1e-4 else f"R={1 / abs(kappa):.0f} m {'right' if kappa > 0 else 'left'}"
    speed_tag = f"speed {ego.speed_kmh:4.1f} km/h" if ego.step >= 0.005 else "speed 0.0 km/h [STOPPED]"
    text = f"{speed_tag}  curvature {kappa * 1000:+.1f} /km  {radius}  corners {len(ego.pts)} (used {ego.n_used})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (10, 10), (30 + tw, 52 + th - 20), (0, 0, 0), -1)
    cv2.putText(img, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def process_frame(frame, ego, show_bev=True, show_corners=False):
    img = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ego.update(gray)
    kappa = ego.kappa

    clean = img.copy() if show_bev else None
    draw_path(img, kappa)
    if show_corners:
        for x, y in ego.pts.reshape(-1, 2):
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), -1)
    if show_bev:
        draw_bev_inset(img, clean, ego, kappa)
    draw_hud(img, ego, kappa)
    return img


def frame_reader(vid, q):
    while True:
        ret, frame = vid.read()
        q.put(frame if ret else None)
        if not ret:
            return


def main():
    vid = cv2.VideoCapture(VIDEO_PATH)
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"Frame count: {frame_count}, FPS: {fps}")

    # Main video frame window
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", *WINDOW_SIZE)

    # Separate real-time LineString pixel route map window
    cv2.namedWindow("Route Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Route Map", *MAP_WINDOW_SIZE)

    q = queue.Queue(maxsize=8)
    threading.Thread(target=frame_reader, args=(vid, q), daemon=True).start()

    ego = EgoMotion(fps)
    route_map = PixelRouteMap(width=MAP_WINDOW_SIZE[0], height=MAP_WINDOW_SIZE[1])

    show_bev, show_corners = True, False
    n = 0
    default_geojson_path = Path(__file__).resolve().parent.parent / "data" / "route.geojson"

    while True:
        frame = q.get()
        if frame is None:
            print(f"Stopped at frame {n} of {frame_count}")
            break
        n += 1

        # Process driving video frame (updates ego speed, yaw, and curvature)
        processed_frame = process_frame(frame, ego, show_bev, show_corners)
        cv2.imshow("Frame", processed_frame)

        # Update and render route map in separate window with dynamic speed
        route_map.update(ego)
        map_img = route_map.render()
        cv2.imshow("Route Map", map_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("b"):
            show_bev = not show_bev
        elif key == ord("c"):
            show_corners = not show_corners
        elif key == ord("r"):
            route_map.reset()
            print("Route reset to origin.")
        elif key == ord("s"):
            saved = route_map.save_geojson(default_geojson_path)
            print(f"Manually saved route LineString feature to: {saved}")

    # Auto-save route LineString on completion
    saved_path = route_map.save_geojson(default_geojson_path)
    feat = route_map.get_feature(coordinate_type="pixel")
    print("\n--- Route Summary ---")
    print(f"Total distance: {route_map.total_dist:.2f} m ({route_map.total_dist / 1000.0:.3f} km)")
    print(f"LineString vertices: {len(feat['geometry']['coordinates'])} points")
    print(f"GeoJSON feature saved to: {saved_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
