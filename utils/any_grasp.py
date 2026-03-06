from types import SimpleNamespace
from gsnet import AnyGrasp  # requires gsnet.so + license in grasp_detection folder

class AnyGraspDetector:
    def __init__(self, checkpoint_path, max_gripper_width=0.08, gripper_height=0.03, top_down_grasp=False, debug=False):
        # demo.py uses argparse cfgs with these fields :contentReference[oaicite:7]{index=7}
        self.cfgs = SimpleNamespace(
            checkpoint_path=checkpoint_path,
            max_gripper_width=max_gripper_width,
            gripper_height=gripper_height,
            top_down_grasp=top_down_grasp,
            debug=debug,
        )
        self.model = AnyGrasp(self.cfgs)
        self.model.load_net()

    def infer(self, points_xyz, colors_rgb=None, lims=None,
              apply_object_mask=True, dense_grasp=False, collision_detection=True):
        pts = np.asarray(points_xyz, dtype=np.float32)

        if colors_rgb is None:
            cols = np.zeros_like(pts, dtype=np.float32)
        else:
            cols = np.asarray(colors_rgb, dtype=np.float32)
            if cols.max() > 1.0:
                cols = cols / 255.0

        gg, cloud = self.model.get_grasp(
            pts, cols,
            lims=lims,
            apply_object_mask=apply_object_mask,
            dense_grasp=dense_grasp,
            collision_detection=collision_detection,
        )  # signature shown in demo :contentReference[oaicite:8]{index=8}

        if len(gg) == 0:
            return gg, cloud

        gg = gg.nms().sort_by_score()  # demo usage :contentReference[oaicite:9]{index=9}
        return gg, cloud

# Initialize once (after env reset, before loop)
det = AnyGraspDetector(
    checkpoint_path="/abs/path/to/checkpoint_detection.tar",  # you pass this in demo :contentReference[oaicite:10]{index=10}
    max_gripper_width=0.08,   # SO101 gripper max opening (meters)
    gripper_height=0.03,
    top_down_grasp=True,      # optional; demo supports it :contentReference[oaicite:11]{index=11}
)

# In your loop, when you have pc:
pc = rigid_object_pc(env, ['Orange001', 'Orange002', 'Orange003'])  # (I’d drop Plate for grasping)

# Example world-frame workspace box (edit to your scene)
xmin, xmax = pc[:,0].min() - 0.10, pc[:,0].max() + 0.10
ymin, ymax = pc[:,1].min() - 0.10, pc[:,1].max() + 0.10
zmin, zmax = pc[:,2].min() - 0.05, pc[:,2].max() + 0.20
lims = [xmin, xmax, ymin, ymax, zmin, zmax]

gg, _ = det.infer(pc, colors_rgb=None, lims=lims)

if len(gg) > 0:
    g0 = gg[0]
    print("best score:", float(g0.score))
    # Next: convert g0 to an EE target pose (see below)