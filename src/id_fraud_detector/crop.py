"""Document localization and rectification with classic OpenCV (no learned model).

Pipeline: coarse crop around the app's visual frame -> edges -> largest 4-sided contour with an
ID-card aspect ratio -> perspective transform to a fixed-size rectangle.
Returns None when no plausible contour is found (the caller sends the photo to manual review).
"""
import cv2
import numpy as np

ID1_ASPECT = 85.6 / 54.0          # ISO/IEC 7810 ID-1 (Colombian cédula)
OUT_W, OUT_H = 1000, 630          # canonical rectified size (aspect ≈ 1.587)
COARSE_MARGIN = 0.20              # extra margin around the frame region before searching
MIN_AREA_FRAC = 0.10              # contour must cover at least this fraction of the coarse crop
ASPECT_TOL = 0.25                 # accepted deviation from ID-1 aspect


def coarse_crop(img: np.ndarray, margin: float = COARSE_MARGIN) -> tuple[np.ndarray, tuple[int, int]]:
    """Keep the central region where the app's frame guide sits, plus a margin."""
    h, w = img.shape[:2]
    x0, x1 = int(w * (0.5 - 0.5 / (1 + margin))), int(w * (0.5 + 0.5 / (1 + margin)))
    y0, y1 = int(h * (0.5 - 0.5 / (1 + margin))), int(h * (0.5 + 0.5 / (1 + margin)))
    return img[y0:y1, x0:x1], (x0, y0)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def find_card_quad(img: np.ndarray) -> np.ndarray | None:
    """Largest 4-sided contour with a card-like aspect ratio, or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = MIN_AREA_FRAC * img.shape[0] * img.shape[1]
    best, best_area = None, 0
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(c)
        if area < area_min:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = order_corners(approx)
        w = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
        h = max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
        if h == 0:
            continue
        aspect = w / h
        if abs(aspect - ID1_ASPECT) / ID1_ASPECT <= ASPECT_TOL and area > best_area:
            best, best_area = quad, area
    return best


def rectify(img: np.ndarray, quad: np.ndarray, size: tuple[int, int] = (OUT_W, OUT_H)) -> np.ndarray:
    dst = np.array([[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, size)


def crop_document(img: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Full pipeline. Returns (rectified_card, quad_in_original_coords) or (None, None)."""
    region, (ox, oy) = coarse_crop(img)
    quad = find_card_quad(region)
    if quad is None:  # retry on the full frame in case the card sits outside the guide
        region, (ox, oy) = img, (0, 0)
        quad = find_card_quad(region)
    if quad is None:
        return None, None
    quad = quad + np.array([ox, oy], dtype="float32")
    return rectify(img, quad), quad
