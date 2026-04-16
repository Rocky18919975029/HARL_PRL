# harl/runners/on_policy_ha_runner.py

"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyHARunner(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms."""

    def train(self):
        """Train the model."""
        actor_train_infos = []

        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())
            
        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(
                factor
            )  # current actor save factor

            # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            # compute action log probs for the actor before update.
            # === 【修改为 (期望 3 个值，因为默认 return_aux=False)】 ===
            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # =========================================================================
            # === 【核心修改：将 Actor 的 mini-batch 更新逻辑移至 Runner】 ===
            # =========================================================================
            
            # Prepare advantages for this agent
            if self.state_type == "EP":
                adv_to_use = advantages.copy()
            elif self.state_type == "FP":
                adv_to_use = advantages[:, :, agent_id].copy()
                
            # Perform EP advantage normalization if needed
            if self.state_type == "EP":
                adv_copy_for_norm = adv_to_use.copy()
                adv_copy_for_norm[self.actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                mean_adv = np.nanmean(adv_copy_for_norm)
                std_adv = np.nanstd(adv_copy_for_norm)
                adv_to_use = (adv_to_use - mean_adv) / (std_adv + 1e-5)

            # Initialize train info
            actor_train_info = {
                "policy_loss": 0, "dist_entropy": 0, 
                "actor_grad_norm": 0, "ratio": 0, "aux_loss": 0,
                # === 【新增】：初始化 action_pred_loss ===
                "action_pred_loss": 0
            }

            # Only train if there are active agents
            if not np.all(self.actor_buffer[agent_id].active_masks[:-1] == 0.0):
                ppo_epoch = self.actor[agent_id].ppo_epoch
                actor_num_mini_batch = self.actor[agent_id].actor_num_mini_batch
                
                use_aux_loss = self.algo_args["algo"].get("use_aux_loss", False)
                use_action_pred = self.algo_args["algo"].get("use_action_pred", False)

                # === 【增量修改 7】：拼接全体智能体的动作 ===
                if use_action_pred:
                    all_actions = [self.actor_buffer[i].actions for i in range(self.num_agents)]
                    joint_actions_input = np.concatenate(all_actions, axis=-1)
                else:
                    joint_actions_input = None

                for _ in range(ppo_epoch):
                    # Call generator
                    # 如果使用 aux_loss，则把 share_obs 传进去，否则传 None
                    share_obs_to_pass = self.critic_buffer.share_obs if use_aux_loss else None
                    
                    if self.actor[agent_id].use_recurrent_policy:
                        data_generator = self.actor_buffer[agent_id].recurrent_generator_actor(
                            adv_to_use, actor_num_mini_batch, self.actor[agent_id].data_chunk_length,
                            share_obs=share_obs_to_pass, joint_actions=joint_actions_input
                        )
                    elif self.actor[agent_id].use_naive_recurrent_policy:
                        data_generator = self.actor_buffer[agent_id].naive_recurrent_generator_actor(
                            adv_to_use, actor_num_mini_batch,
                            share_obs=share_obs_to_pass,
                            joint_actions=joint_actions_input
                        )
                    else:
                        data_generator = self.actor_buffer[agent_id].feed_forward_generator_actor(
                            adv_to_use, actor_num_mini_batch,
                            share_obs=share_obs_to_pass,
                            joint_actions=joint_actions_input
                        )

                    for sample in data_generator:
                        target_embedding = None
                        target_joint_actions = None
                        
                        # 转成 list 以便安全 pop
                        current_sample = list(sample)
                        
                        if use_action_pred:
                            # Buffer 中 joint_actions 是最后(最外层)放进去的，所以第一个 pop 出来
                            target_joint_actions = torch.tensor(current_sample.pop()).to(self.device)
                        
                        if use_aux_loss:
                            # 接着 pop 出来的才是 next_share_obs_batch
                            next_share_obs_batch = current_sample.pop()
                            
                            if next_share_obs_batch is not None:
                                with torch.no_grad():
                                    b_size = next_share_obs_batch.shape[0]
                                    rec_n = self.critic.critic.recurrent_n
                                    rnn_hid_size = self.critic.critic.hidden_sizes[-1]
                                    
                                    dummy_rnn = np.zeros((b_size, rec_n, rnn_hid_size), dtype=np.float32)
                                    dummy_masks = np.ones((b_size, 1), dtype=np.float32)
                                    
                                    target_embedding = self.critic.critic.get_embedding(
                                        next_share_obs_batch, dummy_rnn, dummy_masks
                                    )
                        
                        # 经过上面按条件 pop 之后，剩下的 current_sample 绝对是最纯净的基础 PPO 数据！
                        actor_sample = tuple(current_sample)
                        
                        # --- 调用 update ---
                        policy_loss, dist_entropy, actor_grad_norm, imp_weights, aux_loss, action_pred_loss = self.actor[agent_id].update(
                            actor_sample, target_embedding, target_joint_actions
                        )

                        actor_train_info["policy_loss"] += policy_loss.item()
                        actor_train_info["dist_entropy"] += dist_entropy.item()
                        actor_train_info["actor_grad_norm"] += actor_grad_norm
                        actor_train_info["ratio"] += imp_weights.mean()
                        actor_train_info["aux_loss"] += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
                        actor_train_info["action_pred_loss"] += action_pred_loss.item() if isinstance(action_pred_loss, torch.Tensor) else action_pred_loss

                num_updates = ppo_epoch * actor_num_mini_batch
                for k in actor_train_info.keys():
                    actor_train_info[k] /= num_updates

            # =========================================================================

            # compute action log probs for updated agent
           # === 【修改为 (期望 3 个值，因为默认 return_aux=False)】 ===
            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # update factor for next agent
            factor = factor * _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info