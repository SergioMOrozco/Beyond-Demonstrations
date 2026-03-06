import numpy as np
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R

# Load once

def load_chain():
    chain = Chain.from_urdf_file(
        "robot_urdf/so101_new_calib.urdf",
        # you may need to set these depending on your URDF
        # base_elements=["base_link"],
        # last_link_vector=[0, 0, tcp_offset],
    )

    return chain

def make_T(pos, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T

def solve_ik_frame(chain, pos, quat_xyzw, q_init):
    T = make_T(pos, quat_xyzw)
    q_sol = chain.inverse_kinematics_frame(
        T,
        initial_position=q_init,
    )
    return np.asarray(q_sol, dtype=np.float64)