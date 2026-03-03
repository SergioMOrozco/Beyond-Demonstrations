# train.py
import torch
import torch.nn.functional as F
import cv2

from models.model import WorldModel
#from common.layers import api_model_conversion
from tensordict import TensorDict

from dataset import get_dataset, LiberoWrapper, collate_fn
from torch.utils.data import DataLoader

def _to_device_obs_and_action(batch, device):
    """
    batch from your collate_fn:
      batch["obs"][k]: (B,T,Dk) or (B,Dk)
      batch["actions"]: (B,T,A) or (B,A)
    """
    obs = {k: v.to(device, non_blocking=True).float() for k, v in batch["obs"].items()}
    act = batch["actions"].to(device, non_blocking=True).float()
    return obs, act


def _concat_obs(obs_dict):
    """
    Concatenate your chosen obs into one tensor on the last dimension.

    Returns:
      (B,T,obs_dim) or (B,obs_dim)
    """
    return torch.cat(
        [obs_dict["joint_states"], obs_dict["gripper_states"], obs_dict["target_obj_pos"]],
        dim=-1,
    )


def _bt_to_tb(x: torch.Tensor) -> torch.Tensor:
    """(B,T,D) -> (T,B,D)."""
    return x.permute(1, 0, 2).contiguous()


class ModelTrainer(torch.nn.Module):
    """
    TD-MPC2 agent (model learning portion).
    Adapted to train from a PyTorch DataLoader that yields sequences.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(getattr(cfg, "device", "cuda:0"))
        self.model = WorldModel(cfg).to(self.device)

        self.optim_model = torch.optim.Adam(
            [
                {"params": self.model._encoder.parameters(), "lr": self.cfg.lr * self.cfg.enc_lr_scale},
                {"params": self.model._dynamics.parameters()},
            ],
            lr=self.cfg.lr,
        )

        self.optim_dec = torch.optim.Adam(
            self.model._decoder.parameters(),
            lr=getattr(self.cfg, "dec_lr", 1e-4),
        )

        self.model.eval()

        if getattr(cfg, "compile", False):
            print("Compiling update function with torch.compile...")
            self._update = torch.compile(self._update, mode="reduce-overhead")

    def save(self, fp):
        torch.save({"model": self.model.state_dict()}, fp)

    #def load(self, fp):
    #    if isinstance(fp, dict):
    #        state_dict = fp
    #    else:
    #        state_dict = torch.load(fp, map_location="cpu", weights_only=False)
    #    state_dict = state_dict["model"] if "model" in state_dict else state_dict
    #    state_dict = api_model_conversion(self.model.state_dict(), state_dict)
    #    self.model.load_state_dict(state_dict)

    def _update(self, obs_tbd, action_tba, img_btchw):
        """
        obs_tbd:   (T, B, obs_dim)   where T = horizon+1
        action_tba:(T-1, B, act_dim) where T-1 = horizon

        terminated is unused for now (kept for API compatibility).
        """

        #=======================DYNAMICS LOSS====================
        breakpoint()
        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs_tbd[1:])  # (T-1,B,Z)

        # Prepare for update
        self.model.train()

        # Latent rollout
        Tm1, B, _ = action_tba.shape
        zs = torch.empty(Tm1 + 1, B, self.cfg.latent_dim, device=self.device)

        z = self.model.encode(obs_tbd[0])  # (B,Z)
        zs[0] = z

        consistency_loss = 0.0
        for t, (_action, _next_z) in enumerate(zip(action_tba.unbind(0), next_z.unbind(0))):
            z = self.model.next(z, _action)
            consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * (self.cfg.rho ** t)
            zs[t + 1] = z

        consistency_loss = consistency_loss / float(self.cfg.horizon)

        #total_loss = (self.cfg.consistency_coef * consistency_loss)
        total_loss = consistency_loss

        # Update model
        self.optim_model.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim_model.step()

        #=======================DECODER LOSS====================
        dec_loss = None

        # choose which time(s) to decode; simplest: t=0
        rgb0 = img_btchw[:, 0]  # (B,C,H,W)

        # get latent WITHOUT allowing grads into world model
        with torch.no_grad():
            z0 = self.model.encode(obs_tbd[0])  # (B,Z)

        # decoder forward DOES require grads (decoder params only)
        pred = self.model.decode(z0)         # (B,3,H,W)

        img = rgb0[0].detach().cpu()          # (3,H,W)
        img = img.permute(1,2,0)              # (H,W,3)
        img = img.numpy()

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # normalize
        img = (img * 255).astype("uint8")

        img = img[:,:,::-1]  # RGB → BGR

        img_pred = pred[0].detach().cpu()          # (3,H,W)
        img_pred = img_pred.permute(1,2,0)              # (H,W,3)
        img_pred = img_pred.numpy()

        img_pred = (img_pred - img_pred.min()) / (img_pred.max() - img_pred.min() + 1e-8)  # normalize
        img_pred = (img_pred * 255).astype("uint8")

        img_pred = img_pred[:,:,::-1]  # RGB → BGR

        cv2.imshow("image", img)
        cv2.imshow("pred", img_pred)
        cv2.waitKey(1)

        # pick a recon loss (L1 is a good start)
        dec_loss = F.l1_loss(pred, rgb0)

        self.optim_dec.zero_grad(set_to_none=True)
        dec_loss.backward()
        self.optim_dec.step()

        #===========================================

        # Return training statistics
        self.model.eval()
        info = TensorDict(
            {
                "consistency_loss": consistency_loss.detach(),
                "total_loss": total_loss.detach(),
                "grad_norm": torch.as_tensor(grad_norm).detach(),
                "dec_loss": dec_loss.detach(),
            },
            batch_size=[]

        )
        return info

    def train_one_epoch(self, loader):
        """
        Train over one epoch of DataLoader batches.
        Expects sequences from LiberoWrapper.

        IMPORTANT:
          We assume your samples are (T,*) and batches are (B,T,*)
          We set horizon = T-1 (so obs length is horizon+1).
        """
        all_info = []
        for batch in loader:
            obs_dict, act = _to_device_obs_and_action(batch, self.device)

            rgb = batch["obs"]["rgb_view"].to(self.device, non_blocking=True).float()  # (B,T,C,H,W) or (B,C,H,W)

            # concat obs -> (B,T,obs_dim)
            obs_btD = _concat_obs(obs_dict)

            # Ensure we have sequences
            if obs_btD.ndim != 3:
                raise ValueError(f"Expected obs shape (B,T,D). Got {tuple(obs_btD.shape)}")
            if act.ndim != 3:
                raise ValueError(f"Expected action shape (B,T,A). Got {tuple(act.shape)}")

            B, Tobs, _ = obs_btD.shape
            _, Tact, _ = act.shape

            # We need obs length = horizon+1 and action length = horizon.
            # If actions are stored aligned with obs (Tact == Tobs), drop the last action.
            # If actions are already Tobs-1, keep as-is.
            horizon = self.cfg.horizon
            if Tobs < horizon + 1:
                raise ValueError(f"Need Tobs >= horizon+1={horizon+1}, got {Tobs}")

            # Slice a consistent window from the start
            obs_btD = obs_btD[:, : horizon + 1, :]  # (B,H+1,D)

            if Tact == Tobs:
                act_bta = act[:, :horizon, :]        # (B,H,A)
            else:
                if Tact < horizon:
                    raise ValueError(f"Need Tact >= horizon={horizon}, got {Tact}")
                act_bta = act[:, :horizon, :]        # (B,H,A)

            # Convert to time-major (T,B,*)
            obs_tbd = _bt_to_tb(obs_btD)   # (H+1,B,D)
            act_tba = _bt_to_tb(act_bta)   # (H,B,A)

            if getattr(torch, "compiler", None) is not None:
                torch.compiler.cudagraph_mark_step_begin()

            info = self._update(obs_tbd, act_tba, rgb)
            all_info.append(info)

        # Return mean stats (simple)
        if len(all_info) == 0:
            return {}
        mean_total = torch.stack([x["total_loss"] for x in all_info]).mean().item()
        mean_cons  = torch.stack([x["consistency_loss"] for x in all_info]).mean().item()
        mean_dec  = torch.stack([x["dec_loss"] for x in all_info]).mean().item()
        return {"total_loss": mean_total, "consistency_loss": mean_cons, "decoder loss" : mean_dec}


if __name__ == "__main__":
    # ---- dataset / loader ----
    seq_len = 2
    batch_size = 32

    dataset, shape_meta = get_dataset(
        dataset_path="data/libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5",
        seq_len=seq_len,
        frame_stack=1,
        hdf5_cache_mode="low_dim",
    )

    wrapped = LiberoWrapper(
        dataset,
        target_obj_pos_slice=(10, 17),
    )

    loader = DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # ---- cfg shim (adjust to your cfg system) ----
    class Cfg:
        device = "cuda:0"
        lr = 1e-5
        enc_lr_scale = 0.3
        compile = False
        obs = "state"

        horizon = seq_len - 1          # IMPORTANT
        batch_size = batch_size
        latent_dim = 512
        obs_shape=16
        num_enc_layers=2
        enc_dim=256
        simnorm_dim= 8
        action_dim=7
        mlp_dim=512
        rho = 0.5
        consistency_coef = 20
        grad_clip_norm = 20
        log_std_min = -10
        log_std_max = 2

        num_channels = 32

        rgb_shape = (3, 128, 128)

    cfg = Cfg()
    trainer = ModelTrainer(cfg)

    for i in range(1000):
        stats = trainer.train_one_epoch(loader)
        print("train stats:", stats)
        
    print("Saving model to: model.pth")
    trainer.save("model.pth")