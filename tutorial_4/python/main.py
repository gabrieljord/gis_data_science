"""
Predicted-path overlay that follows the arc of the road.
Keys:  q = quit, c = toggle tracked corners
"""

import queue
import threading

import cv2
import numpy as np

VIDEO_PATH = r""

W, H = 1920, 1080
WINDOW_SIZE = (1280, 720)


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

ASSUMED_SPEED_MPS = 15.0              
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
        self.step = ASSUMED_SPEED_MPS / fps             
        self.n_used = 0

    @property
    def kappa(self):
        """Path curvature in 1/m (+ = curving right)."""
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
        if len(self.pts) < MIN_TRACKED:
            self._detect(crop)
        self.prev_crop = crop

    def _estimate(self, p0, p1):
        g0, g1 = transform(p0, I2G), transform(p1, I2G)
        z0 = g0[:, 1]
        ok = (z0 > Z_NEAR * 0.9) & (z0 < Z_TRACK_FAR * 1.1) & (g1[:, 1] > 1)
        self.n_used = int(ok.sum())
        if self.n_used < MIN_USED:
            return
        g0, g1 = g0[ok], g1[ok]
        psi = np.median(-(g1[:, 0] - g0[:, 0]) / g0[:, 1])
        self.yaw += YAW_EMA * (psi - self.yaw)

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
    text = f"curvature {kappa * 1000:+.1f} /km  {radius}  corners {len(ego.pts)} (used {ego.n_used})"
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

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", *WINDOW_SIZE)

    q = queue.Queue(maxsize=8)
    threading.Thread(target=frame_reader, args=(vid, q), daemon=True).start()

    ego = EgoMotion(fps)
    show_bev, show_corners = True, False
    n = 0
    while True:
        frame = q.get()
        if frame is None:
            print(f"Stopped at frame {n} of {frame_count}")
            break
        n += 1

        cv2.imshow("Frame", process_frame(frame, ego, show_bev, show_corners))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("b"):
            show_bev = not show_bev
        elif key == ord("c"):
            show_corners = not show_corners

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
