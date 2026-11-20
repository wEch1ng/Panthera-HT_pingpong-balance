"""RGB-D sphere fit and short ballistic tracking; no simulator ball state input.

Depth must be registered to RGB, in metres along the camera optical axis.
Camera transforms use MuJoCo's +X right, +Y up, -Z forward convention.
The known 20 mm ball radius is a task prior, not a ground-truth position.
"""
import cv2
import numpy as np
from perception.detector import mask


def locate_ball(rgb, depth, fovy, radius=.02):
    height, width = depth.shape
    focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask(rgb), 8)
    candidates = []
    for k in range(1, count):
        _, _, bw, bh, area = stats[k]
        if area < 15 or max(bw, bh) > 1.8 * min(bw, bh) or area / (bw*bh) < .4:
            continue
        interior = cv2.erode((labels == k).astype(np.uint8), np.ones((3,3),np.uint8))
        yy, xx = np.where(interior)
        z = depth[yy, xx]
        ok = np.isfinite(z) & (z > .04) & (z < 1.2)
        xx, yy, z = xx[ok], yy[ok], z[ok]
        if len(z) < 12:
            continue
        # RGB antialiasing can mark background at the sphere silhouette.
        # Keep a coherent depth layer before fitting the curved surface.
        ok = np.abs(z-np.median(z)) < 1.5*radius
        xx, yy, z = xx[ok], yy[ok], z[ok]
        if len(z) < 12:
            continue
        points = np.column_stack(((xx + .5 - width/2)*z/focal,
                                  -(yy + .5 - height/2)*z/focal, -z))
        # Translate before fitting to avoid poor conditioning at large depths.
        origin = points.mean(0)
        rel = points - origin
        fit, _, rank, _ = np.linalg.lstsq(
            np.column_stack((2*rel, np.ones(len(rel)))), (rel*rel).sum(1), rcond=None)
        center = origin + fit[:3]
        fitted_radius = np.sqrt(max(0., fit[3] + fit[:3] @ fit[:3]))
        residual = np.median(np.abs(np.linalg.norm(points-center, axis=1) - radius))
        if rank < 4 or abs(fitted_radius-radius) > .006 or residual > .004:
            continue
        cx, cy = centers[k]
        det = np.array([(cx-width/2)/(width/2), (cy-height/2)/(height/2),
                        np.sqrt(area/np.pi)/10, 1.], np.float32)
        candidates.append((residual, center, det))
    if not candidates:
        return None, np.zeros(4, np.float32)
    _, center, det = min(candidates, key=lambda item: item[0])
    return center, det


class RGBDBallTracker:
    def __init__(self, dt=1/30):
        self.dt = dt
        self.reset()

    def reset(self):
        self.position = None
        self.velocity = np.zeros(3)
        self.age = 1.
        self.measurement = None
        self.measurement_age = 0.

    def update(self, point_world):
        self.measurement_age += self.dt
        if point_world is not None:
            if self.measurement is not None and self.age < .25:
                raw = (point_world - self.measurement) / self.measurement_age
                self.velocity = .5*self.velocity + .5*np.clip(raw, -3., 3.)
            else:
                self.velocity[:] = 0
            self.position = point_world.copy()
            self.measurement = point_world.copy()
            self.measurement_age = 0.
            self.age = 0.
        else:
            # Extrapolate only for brief occlusion. Age/visibility are policy inputs.
            if self.position is not None and self.age < .2:
                self.position += self.velocity*self.dt + np.array([0, 0, -4.905*self.dt**2])
                self.velocity[2] -= 9.81*self.dt
            self.age += self.dt
        return self.position, self.velocity, float(point_world is not None), self.age
