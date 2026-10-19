"""CPU tests of the paper's gradient boundary, losses, and target alignment."""
import json
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.mappo import MAPPO
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.models.value_function_models.v_net import VNet

ROOT = Path(__file__).resolve().parents[1]

# HARL uses these space names/shapes. Minimal spaces keep the core tests
# independent of benchmark engines and the legacy Gym installation.
class Box:
    def __init__(self, shape):
        self.shape = shape


class Discrete:
    def __init__(self, n):
        self.n = n


def settings():
    cfg = json.loads((ROOT / "tuned_configs/football/academy_3_vs_1_with_keeper/prl/config.json").read_text())
    args = {**cfg["algo_args"]["model"], **cfg["algo_args"]["algo"], **cfg["algo_args"]["train"]}
    args.update(hidden_sizes=[16, 16], num_res_blocks=1, episode_length=4, n_rollout_threads=2)
    return args


class PRLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        np.random.seed(7)

    def test_actor_aux_update_stops_critic_gradient(self):
        for actor_class in (HAPPO, MAPPO):
            for space in (Discrete(5), Box((2,))):
                with self.subTest(actor=actor_class.__name__, space=space.__class__.__name__):
                    args = settings()
                    actor = actor_class(args, Box((7,)), space)
                    critic = VNet(args, Box((11,)))
                    obs, state = torch.randn(8, 7), torch.randn(8, 11)
                    rnn, masks = torch.zeros(8, 1, 16), torch.ones(8, 1)
                    with torch.no_grad():
                        action, old_logp, _ = actor.actor(obs, rnn, masks)
                    target = critic.get_embedding(state, rnn, masks)
                    self.assertTrue(target.requires_grad)
                    active = masks.clone()
                    active[-2:] = 0
                    sample = (obs, rnn, action, masks, active, old_logp, torch.ones(8, 1), None)
                    if actor_class is HAPPO:
                        sample += (torch.ones(8, 1),)
                    before = actor.actor.aux_head[-1].weight.detach().clone()
                    with torch.no_grad():
                        pred = actor.actor.evaluate_actions(obs, rnn, action, masks, None, active, return_aux=True)[3]
                        expected = (F.mse_loss(pred, F.layer_norm(target, [16]), reduction="none").mean(-1, keepdim=True) * active).sum() / active.sum()
                    result = actor.update(sample, target_embedding=target)
                    aux = result[4] if actor_class is HAPPO else result[5]
                    self.assertTrue(torch.isfinite(aux))
                    torch.testing.assert_close(aux, expected)
                    self.assertFalse(torch.equal(before, actor.actor.aux_head[-1].weight))
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.actor.base.parameters()))
                    self.assertTrue(all(p.grad is None for p in critic.parameters()))
                    self.assertEqual(action.shape, (8, 1 if space.__class__.__name__ == "Discrete" else 2))

    def test_disabled_auxiliary_and_joint_action_updates(self):
        for actor_class in (HAPPO, MAPPO):
            for joint in (False, True):
                with self.subTest(actor=actor_class.__name__, joint=joint):
                    args = settings()
                    args.update(use_aux_loss=False, use_action_pred=joint, total_action_dim=6)
                    actor = actor_class(args, Box((7,)), Discrete(5))
                    obs, rnn, masks = torch.randn(8, 7), torch.zeros(8, 1, 16), torch.ones(8, 1)
                    with torch.no_grad():
                        action, old_logp, _ = actor.actor(obs, rnn, masks)
                    sample = (obs, rnn, action, masks, masks, old_logp, torch.ones(8, 1), None)
                    if actor_class is HAPPO:
                        sample += (masks,)
                    target = torch.randn(8, 6, requires_grad=True) if joint else None
                    result = actor.update(sample, target_joint_actions=target)
                    self.assertTrue(all(torch.isfinite(value).all() for value in result))
                    if joint:
                        self.assertIsNone(target.grad)
                        self.assertIsNotNone(actor.actor.action_pred_head[-1].weight.grad)

    def test_buffer_keeps_current_next_targets_aligned(self):
        args = settings()
        buffer = OnPolicyActorBuffer(args, Box((1,)), Discrete(5))
        time = np.arange(5, dtype=np.float32)[:, None, None]
        buffer.obs[:] = time
        share_obs = np.broadcast_to(time * 10, (5, 2, 1)).copy()
        advantages = np.ones((4, 2, 1), dtype=np.float32)
        for target, shift in (("current", 0), ("next", 1)):
            batches = buffer.feed_forward_generator_actor(advantages, 2, share_obs=share_obs, prl_target=target)
            for batch in batches:
                np.testing.assert_allclose(batch[-1], (batch[0] + shift) * 10)


if __name__ == "__main__":
    unittest.main()
