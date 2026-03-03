import numpy as np
import torch
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from torch.utils.data import Dataset, DataLoader
from robomimic.utils.dataset import SequenceDataset

#
#"""
#    Helper function from Robomimic to read hdf5 demonstrations into sequence dataset
#
#    ISSUE: robomimic's SequenceDataset has two properties: seq_len and frame_stack,
#    we should in principle use seq_len, but the paddings of the two are different.
#    So that's why we currently use frame_stack instead of seq_len.
#"""
#
#

def get_dataset(
    dataset_path,
    seq_len=10,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    *args,
    **kwargs
):

    obs_modality = {
        "rgb": ["agentview_rgb", 'eye_in_hand_rgb'],
        "depth": [],
        "low_dim": ["gripper_states", "joint_states"]
    }

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )

    seq_len = seq_len
    filter_key = filter_key
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=shape_meta["all_obs_keys"],
        dataset_keys=["actions", "states"],
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
    )

    return dataset, shape_meta


# -----------------------------
# Wrap robomimic SequenceDataset
# -----------------------------
class LiberoWrapper(Dataset):
    """
    Wraps a robomimic SequenceDataset (or any dataset that returns dicts like you showed)
    and exposes only:
      obs: joint_states, gripper_states, target_obj_pos
      actions
    """

    def __init__(
        self,
        base_dataset,
        target_obj_pos_slice=(10, 17),   # states[0][10:17]
    ):
        self.ds = base_dataset
        self.s0, self.s1 = target_obj_pos_slice

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        # robomimic SequenceDataset returns sequences: shape (seq_len, dim)
        joint = item["obs"]["joint_states"]          # (T, 7) or (7,) depending on config
        grip  = item["obs"]["gripper_states"]        # (T, G)
        img = item['obs']['agentview_rgb']
        act   = item["actions"]                      # (T, A) or (A,)
        st    = item["states"]                       # (T, 110) or (110,)

        # Ensure numpy arrays
        joint = np.asarray(joint)
        grip  = np.asarray(grip)
        img  = np.asarray(img)
        act   = np.asarray(act)
        st    = np.asarray(st)

        # Target object position lives in your "states" vector, but remember:
        # states[0] has a leading scalar at index 0, then qpos+qvel
        # You empirically found target at [10:17] on the *single-timestep* vector.
        # If we're still sequence-shaped, slice per timestep.
        if st.ndim == 1:
            target_obj_pos = st[self.s0:self.s1]              # (7,)
        else:
            target_obj_pos = st[:, self.s0:self.s1]           # (T, 7)

        obs = {
            "joint_states": torch.from_numpy(joint).float(),
            "gripper_states": torch.from_numpy(grip).float(),
            "target_obj_pos": torch.from_numpy(target_obj_pos).float(),
            "rgb_view": torch.from_numpy(img).float(),
        }

        actions = torch.from_numpy(act).float()

        return {"obs": obs, "actions": actions}


def collate_fn(batch):
    """
    Collate into:
      obs: dict of tensors stacked on dim 0
      actions: tensor stacked on dim 0
    Works whether tensors are (dim,) or (T, dim).
    """
    out = {"obs": {}, "actions": None}

    # stack each obs key
    obs_keys = batch[0]["obs"].keys()
    for k in obs_keys:
        out["obs"][k] = torch.stack([b["obs"][k] for b in batch], dim=0)

    out["actions"] = torch.stack([b["actions"] for b in batch], dim=0)
    return out

if __name__ == "__main__":
    dataset, shape_meta = get_dataset(
        dataset_path="data/libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5",
        seq_len=10,
        frame_stack=1,
        hdf5_cache_mode="low_dim",
    )

    wrapped = LiberoWrapper(
        dataset,
        target_obj_pos_slice=(10, 17),
    )

    loader = DataLoader(
        wrapped,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    for batch in loader:
        print("obs.joint_states:", batch["obs"]["joint_states"].shape)      # (B, 7)
        print("obs.gripper_states:", batch["obs"]["gripper_states"].shape)  # (B, G)
        print("obs.target_obj_pos:", batch["obs"]["target_obj_pos"].shape)  # (B, 7)
        break