import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_models import (
    DCT_Condition_Module,
    LabelSmoothingBCE,
    MultiModalPromptLearner,
    load_clip_to_cpu,
)
from utils.cie_transforms import (
    build_artifact_view,
    build_patch_family_views,
    build_structure_view,
)


class CounterfactualReliabilityGate(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gate_features):
        gate_logits = self.scorer(gate_features).squeeze(-1)
        return F.softmax(gate_logits, dim=1), gate_logits


class CIEIAPLModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.cie_num_specialists != 3:
            raise ValueError("CIE-IAPL v1.1 implements exactly 3 specialist experts.")

        self.args = args
        self.num_specialists = args.cie_num_specialists
        self.current_stage = "init"
        self._last_logged_stage = None

        cfg = {
            'N_CTX': args.n_ctx,
            'PROMPT_DEPTH': args.prompt_depth,
            'SIZE': [args.image_size, args.image_size],
            'VISION_WIDTH': args.vision_width,
        }

        print(f"[CLIP_LOAD] clip_path={args.clip_path}")
        clip_model = load_clip_to_cpu(
            args.clip_path,
            cfg['N_CTX'],
            args.vit_adapter_list,
            args.text_adapter_list,
            args.prompt_depth,
            args.gate,
        )

        self.prompt_learner = MultiModalPromptLearner(cfg, clip_model)
        self.fc_binary = nn.Linear(768, 1)
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype

        if args.condition:
            self.conditional_ctx = DCT_Condition_Module()
        else:
            self.conditional_ctx = None

        self.delta_ctx = nn.Parameter(torch.empty(self.num_specialists, args.n_ctx, args.vision_width))
        self.delta_deep = nn.ParameterList([
            nn.Parameter(torch.empty(self.num_specialists, args.n_ctx, args.vision_width))
            for _ in range(max(args.prompt_depth - 1, 0))
        ])
        self.specialist_heads = nn.ModuleList([nn.Linear(768, 1) for _ in range(self.num_specialists)])
        self.gate = CounterfactualReliabilityGate(args.cie_gate_hidden_dim)

        self._init_residual_prompts(args.cie_residual_init_std)
        self._clone_base_head_to_specialists()
        self._set_initial_trainable_params()

        self.criterion_weight_dict = {
            "loss_final": 1.0,
            "loss_base": args.cie_lambda_base,
            "loss_spec": args.cie_lambda_spec,
            "loss_gate": args.cie_lambda_gate,
            "loss_oracle": args.cie_lambda_oracle,
            "loss_div": args.cie_lambda_div,
            "loss_patch": args.cie_lambda_patch,
            "loss_fake": args.cie_lambda_fake,
            "loss_family_margin": args.cie_lambda_family_margin,
        }

        if args.smooth:
            self.loss_fn = LabelSmoothingBCE()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        print(f"[CIE_IAPL] eval_mode={args.cie_eval_mode}")
        print(f"[CIE_IAPL] gate_mode={args.cie_gate_mode}")
        print(f"[CIE_IAPL] init_from_iapl_ckpt={args.cie_init_from_iapl_ckpt}")
        print(f"[CIE_IAPL] tile_grid={args.cie_patch_tile_grid}")

        if args.cie_init_from_iapl_ckpt:
            self.initialize_from_iapl_checkpoint(args.cie_init_from_iapl_ckpt)

    def _init_residual_prompts(self, std):
        nn.init.normal_(self.delta_ctx, std=std)
        for delta in self.delta_deep:
            nn.init.normal_(delta, std=std)

    def _clone_base_head_to_specialists(self):
        for head in self.specialist_heads:
            head.load_state_dict(self.fc_binary.state_dict())

    def _set_initial_trainable_params(self):
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = ("adapter" in name or "gamma" in name)

        for param in self.prompt_learner.parameters():
            param.requires_grad = True
        if self.conditional_ctx is not None:
            for param in self.conditional_ctx.parameters():
                param.requires_grad = True
        for param in self.fc_binary.parameters():
            param.requires_grad = True
        self.delta_ctx.requires_grad = True
        for delta in self.delta_deep:
            delta.requires_grad = True
        for head in self.specialist_heads:
            for param in head.parameters():
                param.requires_grad = True
        for param in self.gate.parameters():
            param.requires_grad = self.args.cie_gate_mode == "learned"

    def initialize_from_iapl_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }

        own_state = self.state_dict()
        loadable = {}
        skipped = []
        for key, value in state_dict.items():
            if key in own_state and own_state[key].shape == value.shape:
                loadable[key] = value
            else:
                skipped.append(key)

        missing, unexpected = self.load_state_dict(loadable, strict=False)
        self._clone_base_head_to_specialists()
        print(
            "[CIE_IAPL] loaded_iapl_keys={} skipped_keys={} missing_after_partial_load={} unexpected={}".format(
                len(loadable), len(skipped), len(missing), len(unexpected)
            )
        )
        print("[CIE_IAPL] cloned base head into specialist heads")

    def _set_image_encoder_adapter_trainable(self, trainable):
        for name, param in self.image_encoder.named_parameters():
            if "adapter" in name or "gamma" in name:
                param.requires_grad = trainable
            else:
                param.requires_grad = False

    def _set_shared_trainable(self, trainable):
        for param in self.prompt_learner.parameters():
            param.requires_grad = trainable
        if self.conditional_ctx is not None:
            for param in self.conditional_ctx.parameters():
                param.requires_grad = trainable
        self._set_image_encoder_adapter_trainable(trainable)
        for param in self.fc_binary.parameters():
            param.requires_grad = trainable

    def _set_specialists_trainable(self, trainable):
        self.delta_ctx.requires_grad = trainable
        for delta in self.delta_deep:
            delta.requires_grad = trainable
        for head in self.specialist_heads:
            for param in head.parameters():
                param.requires_grad = trainable

    def _set_gate_trainable(self, trainable):
        trainable = trainable and self.args.cie_gate_mode == "learned"
        for param in self.gate.parameters():
            param.requires_grad = trainable

    def set_epoch(self, epoch):
        if epoch < self.args.cie_warmup_epochs:
            stage = "warmup"
            self._set_shared_trainable(False)
            self._set_specialists_trainable(True)
            self._set_gate_trainable(False)
        elif epoch < self.args.cie_warmup_epochs + self.args.cie_gate_warmup_epochs:
            stage = "gate_warmup"
            self._set_shared_trainable(False)
            self._set_specialists_trainable(True)
            self._set_gate_trainable(True)
        else:
            stage = "joint"
            self._set_shared_trainable(True)
            self._set_specialists_trainable(True)
            self._set_gate_trainable(True)

        self.current_stage = stage
        if stage != self._last_logged_stage:
            print(f"[CIE_IAPL] stage={stage}")
            self._last_logged_stage = stage

    def _base_prompt_parts(self):
        return self.prompt_learner()

    def _make_deep_prompts(self, base_deep_prompts, specialist_idx=None):
        if specialist_idx is None:
            return base_deep_prompts
        return [
            prompt + self.delta_deep[layer_idx][specialist_idx]
            for layer_idx, prompt in enumerate(base_deep_prompts)
        ]

    def _make_ctx(self, base_ctx, condition_images, specialist_idx=None):
        if base_ctx is None:
            return None, None

        ctx = base_ctx
        if specialist_idx is not None:
            ctx = ctx + self.delta_ctx[specialist_idx]

        ctx = ctx.expand(condition_images.shape[0], -1, -1)
        pred_bias = None
        if self.conditional_ctx is not None:
            bias, pred_bias = self.conditional_ctx(condition_images.type(self.dtype))
            ctx = ctx + bias
        return ctx, pred_bias

    def _encode_with_prompts(self, images, ctx, deep_prompts, head):
        image_features, _ = self.image_encoder(images.type(self.dtype), ctx, deep_prompts)
        logits = head(image_features).squeeze(-1)
        return logits, image_features

    def _run_base_expert(self, images, base_ctx, base_deep_prompts):
        ctx, _ = self._make_ctx(base_ctx, images, specialist_idx=None)
        return self._encode_with_prompts(images, ctx, base_deep_prompts, self.fc_binary)

    def _run_specialist_expert(self, images, view_images, base_ctx, base_deep_prompts, specialist_idx):
        condition_images = images if self.args.cie_condition_source == "original" else view_images
        ctx, _ = self._make_ctx(base_ctx, condition_images, specialist_idx=specialist_idx)
        deep_prompts = self._make_deep_prompts(base_deep_prompts, specialist_idx=specialist_idx)
        return self._encode_with_prompts(
            view_images,
            ctx,
            deep_prompts,
            self.specialist_heads[specialist_idx],
        )

    def _run_base_patch_family(self, tile_images, tile_count, batch_size, base_ctx, base_deep_prompts):
        ctx, _ = self._make_ctx(base_ctx, tile_images, specialist_idx=None)
        tile_logits, tile_features = self._encode_with_prompts(
            tile_images,
            ctx,
            base_deep_prompts,
            self.fc_binary,
        )
        tile_logits = tile_logits.view(batch_size, tile_count)
        return tile_logits.mean(dim=1), tile_features, tile_logits

    def _run_patch_expert(self, images, tile_images, tile_count, base_ctx, base_deep_prompts):
        if self.args.cie_condition_source == "original":
            ctx, _ = self._make_ctx(base_ctx, images, specialist_idx=2)
            if ctx is not None:
                ctx = ctx.repeat_interleave(tile_count, dim=0)
        else:
            ctx, _ = self._make_ctx(base_ctx, tile_images, specialist_idx=2)
        deep_prompts = self._make_deep_prompts(base_deep_prompts, specialist_idx=2)
        tile_logits, tile_features = self._encode_with_prompts(
            tile_images,
            ctx,
            deep_prompts,
            self.specialist_heads[2],
        )
        batch_size = images.shape[0]
        tile_logits = tile_logits.view(batch_size, tile_count)
        patch_logit = tile_logits.mean(dim=1)
        alpha = F.softmax(tile_logits / max(self.args.cie_patch_temp, 1e-6), dim=1)
        tile_entropy = -(alpha * torch.log(alpha.clamp_min(1e-8))).sum(dim=1)
        tile_variance = tile_logits.var(dim=1, unbiased=False)
        tile_max = tile_logits.max(dim=1).values
        tile_min = tile_logits.min(dim=1).values
        return patch_logit, tile_features, tile_logits, tile_entropy, tile_variance, tile_max, tile_min

    def _uniform_gate_probs(self, all_logits):
        gate_probs = all_logits.new_full(all_logits.shape, 1.0 / all_logits.shape[1])
        if not self.args.cie_use_base_expert:
            gate_probs[:, 0] = 0.0
            gate_probs[:, 1:] = 1.0 / (all_logits.shape[1] - 1)
        return gate_probs

    def _build_gate_features(self, all_logits, base_family_logits, tile_entropy, tile_variance):
        confidence = torch.maximum(all_logits.sigmoid(), 1.0 - all_logits.sigmoid())
        family_delta = all_logits.new_zeros(all_logits.shape)
        if self.args.cie_use_family_base_refs:
            family_delta[:, 1] = all_logits[:, 1] - base_family_logits["artifact"]
            family_delta[:, 2] = all_logits[:, 2] - base_family_logits["structure"]
            family_delta[:, 3] = all_logits[:, 3] - base_family_logits["patch"]
        else:
            family_delta = all_logits - all_logits[:, :1]
            family_delta[:, 0] = 0.0

        aux = all_logits.new_zeros(all_logits.shape)
        if self.args.cie_gate_use_aux_stat:
            aux[:, 3] = tile_entropy
            aux[:, 3] = aux[:, 3] + 0.0 * tile_variance

        return torch.stack([all_logits, confidence, family_delta, aux], dim=-1), family_delta

    def _gate_probs(self, all_logits, gate_features):
        if self.args.cie_gate_mode == "uniform":
            return self._uniform_gate_probs(all_logits), all_logits.new_zeros(all_logits.shape)
        if self.training and self.current_stage == "warmup":
            return self._uniform_gate_probs(all_logits), all_logits.new_zeros(all_logits.shape)

        gate_probs, gate_logits = self.gate(gate_features)
        if not self.args.cie_use_base_expert:
            gate_probs = gate_probs.clone()
            gate_probs[:, 0] = 0.0
            gate_probs = gate_probs / gate_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return gate_probs, gate_logits

    def _select_eval_logits(self, outputs):
        mode = self.args.cie_eval_mode
        if mode == "final":
            return outputs["final_logit"]
        if mode == "base":
            return outputs["base_logit"]
        if mode == "artifact":
            return outputs["artifact_logit"]
        if mode == "structure":
            return outputs["structure_logit"]
        if mode == "patch":
            return outputs["patch_logit"]
        if mode == "uniform":
            return outputs["all_logits"].mean(dim=1)
        raise ValueError(f"Unsupported cie_eval_mode: {mode}")

    def _forward_impl(self, images):
        base_ctx, base_deep_prompts = self._base_prompt_parts()

        base_logit, _ = self._run_base_expert(images, base_ctx, base_deep_prompts)

        artifact_images = build_artifact_view(
            images,
            mode=self.args.cie_artifact_train_mode,
            training=self.training,
        )
        structure_images = build_structure_view(
            images,
            mode=self.args.cie_structure_train_mode,
            training=self.training,
        )
        patch_tile_images, tile_count, tile_family_name = build_patch_family_views(
            images,
            mode=self.args.cie_patch_train_mode,
            training=self.training,
            grid=self.args.cie_patch_tile_grid,
            drop_prob=self.args.cie_patch_dropout_prob,
            mask_ratio=self.args.cie_patch_mask_ratio,
        )

        base_artifact_logit, _ = self._run_base_expert(artifact_images, base_ctx, base_deep_prompts)
        base_structure_logit, _ = self._run_base_expert(structure_images, base_ctx, base_deep_prompts)
        base_patch_logit, _, base_tile_logits = self._run_base_patch_family(
            patch_tile_images,
            tile_count,
            images.shape[0],
            base_ctx,
            base_deep_prompts,
        )

        artifact_logit, _ = self._run_specialist_expert(
            images,
            artifact_images,
            base_ctx,
            base_deep_prompts,
            specialist_idx=0,
        )
        structure_logit, _ = self._run_specialist_expert(
            images,
            structure_images,
            base_ctx,
            base_deep_prompts,
            specialist_idx=1,
        )
        (
            patch_logit,
            _,
            tile_logits,
            tile_entropy,
            tile_variance,
            tile_max,
            tile_min,
        ) = self._run_patch_expert(
            images,
            patch_tile_images,
            tile_count,
            base_ctx,
            base_deep_prompts,
        )

        expert_logits = torch.stack([artifact_logit, structure_logit, patch_logit], dim=1)
        all_logits = torch.stack([base_logit, artifact_logit, structure_logit, patch_logit], dim=1)
        base_family_logits = {
            "orig": base_logit,
            "artifact": base_artifact_logit,
            "structure": base_structure_logit,
            "patch": base_patch_logit,
        }
        gate_features, family_delta = self._build_gate_features(
            all_logits,
            base_family_logits,
            tile_entropy,
            tile_variance,
        )
        gate_probs, gate_logits = self._gate_probs(all_logits, gate_features)
        final_logit = (gate_probs * all_logits).sum(dim=1)
        uniform_logit = all_logits.mean(dim=1)

        outputs = {
            "final_logit": final_logit,
            "uniform_logit": uniform_logit,
            "base_logit": base_logit,
            "artifact_logit": artifact_logit,
            "structure_logit": structure_logit,
            "patch_logit": patch_logit,
            "expert_logits": expert_logits,
            "all_logits": all_logits,
            "gate_probs": gate_probs,
            "gate_logits": gate_logits,
            "gate_features": gate_features,
            "family_delta": family_delta,
            "base_family_logits": base_family_logits,
            "base_tile_logits": base_tile_logits,
            "tile_logits": tile_logits,
            "tile_entropy": tile_entropy,
            "tile_variance": tile_variance,
            "tile_max": tile_max,
            "tile_min": tile_min,
            "tile_family_name": tile_family_name,
            "patch_family_name": tile_family_name,
            "stage": self.current_stage,
        }

        if self.args.cie_debug_log:
            z_means = all_logits.detach().mean(dim=0).cpu().tolist()
            q_means = gate_probs.detach().mean(dim=0).cpu().tolist()
            d_means = family_delta.detach().mean(dim=0).cpu().tolist()
            print(
                "[CIE_IAPL] patch_family={} mean_z: base/art/str/patch={:.4f}/{:.4f}/{:.4f}/{:.4f} "
                "mean_q: base/art/str/patch={:.4f}/{:.4f}/{:.4f}/{:.4f} "
                "mean_family_delta: art/str/patch={:.4f}/{:.4f}/{:.4f} tile_entropy_mean={:.4f}".format(
                    tile_family_name,
                    z_means[0], z_means[1], z_means[2], z_means[3],
                    q_means[0], q_means[1], q_means[2], q_means[3],
                    d_means[1], d_means[2], d_means[3],
                    tile_entropy.detach().mean().item(),
                )
            )

        return outputs

    def forward_debug(self, images):
        return self._forward_impl(images)

    def forward(self, images):
        outputs = self._forward_impl(images)
        if self.training:
            return outputs
        return self._select_eval_logits(outputs)

    def _loss_diversity(self, residuals):
        if residuals.shape[0] < 2:
            return residuals.sum() * 0.0
        residuals = residuals - residuals.mean(dim=0, keepdim=True)
        residuals = residuals / residuals.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        corr = residuals.transpose(0, 1) @ residuals / residuals.shape[0]
        off_diag = corr - torch.diag(torch.diag(corr))
        return off_diag.pow(2).sum() / 6.0

    def _family_margin_loss(self, spec_logit, base_logit, targets, margin):
        spec_risk = F.binary_cross_entropy_with_logits(spec_logit, targets, reduction="none")
        base_risk = F.binary_cross_entropy_with_logits(base_logit, targets, reduction="none")
        return F.relu(margin + spec_risk - base_risk), spec_risk, base_risk

    def get_criterion(self, outputs, targets):
        targets = targets.float().view(-1)
        final_logit = outputs["final_logit"]
        base_logit = outputs["base_logit"]
        artifact_logit = outputs["artifact_logit"]
        structure_logit = outputs["structure_logit"]
        patch_logit = outputs["patch_logit"]
        expert_logits = outputs["expert_logits"]
        all_logits = outputs["all_logits"]
        gate_probs = outputs["gate_probs"]
        base_family_logits = outputs["base_family_logits"]
        family_delta = outputs["family_delta"]
        tile_logits = outputs["tile_logits"]
        tile_entropy = outputs["tile_entropy"]
        tile_variance = outputs["tile_variance"]

        target_experts = targets.unsqueeze(1).expand_as(expert_logits)
        target_all = targets.unsqueeze(1).expand_as(all_logits)
        risks = F.binary_cross_entropy_with_logits(all_logits, target_all, reduction="none")
        zero = all_logits.sum() * 0.0

        loss_dict = {
            "loss_final": self.loss_fn(final_logit, targets),
            "loss_base": self.loss_fn(base_logit, targets),
            "loss_spec": self.loss_fn(expert_logits, target_experts),
        }

        if self.args.cie_gate_mode == "learned":
            oracle_target = F.softmax(-risks.detach() / max(self.args.cie_oracle_tau, 1e-6), dim=1)
            log_oracle = torch.log(oracle_target.clamp_min(1e-8))
            log_gate = torch.log(gate_probs.clamp_min(1e-8))
            loss_dict["loss_gate"] = (oracle_target * (log_oracle - log_gate)).sum(dim=1).mean()
        else:
            loss_dict["loss_gate"] = zero

        best_risk = risks.min(dim=1).values
        loss_dict["loss_oracle"] = F.relu(
            self.args.cie_oracle_margin + best_risk - risks[:, 0]
        ).mean()

        residuals = torch.stack(
            [
                artifact_logit - base_family_logits["artifact"],
                structure_logit - base_family_logits["structure"],
                patch_logit - base_family_logits["patch"],
            ],
            dim=1,
        )
        loss_dict["loss_div"] = self._loss_diversity(residuals)

        art_margin, _, _ = self._family_margin_loss(
            artifact_logit,
            base_family_logits["artifact"],
            targets,
            self.args.cie_art_margin,
        )
        structure_margin, _, _ = self._family_margin_loss(
            structure_logit,
            base_family_logits["structure"],
            targets,
            self.args.cie_structure_margin,
        )
        patch_margin, _, _ = self._family_margin_loss(
            patch_logit,
            base_family_logits["patch"],
            targets,
            self.args.cie_patch_margin,
        )
        loss_dict["loss_family_margin"] = torch.stack(
            [art_margin, structure_margin, patch_margin],
            dim=1,
        ).mean()

        alpha = F.softmax(tile_logits / max(self.args.cie_patch_temp, 1e-6), dim=1)
        entropy = -(alpha * torch.log(alpha.clamp_min(1e-8))).sum(dim=1)
        fake_mask = targets == 1
        if fake_mask.any():
            loss_dict["loss_patch"] = F.relu(
                self.args.cie_patch_entropy_min - entropy[fake_mask]
            ).mean()
        else:
            loss_dict["loss_patch"] = zero

        hard_fake_mask = (targets == 1) & (base_logit < self.args.cie_hard_fake_threshold)
        correction_orig = expert_logits - base_logit.unsqueeze(1)
        correction_family = torch.stack(
            [
                artifact_logit - base_family_logits["artifact"],
                structure_logit - base_family_logits["structure"],
                patch_logit - base_family_logits["patch"],
            ],
            dim=1,
        )
        if hard_fake_mask.any():
            best_correction = correction_orig[hard_fake_mask].max(dim=1).values
            loss_dict["loss_fake"] = F.relu(
                self.args.cie_hard_fake_margin - best_correction
            ).mean()
        else:
            loss_dict["loss_fake"] = zero

        gate_entropy = -(gate_probs * torch.log(gate_probs.clamp_min(1e-8))).sum(dim=1)
        oracle_match = (gate_probs.argmax(dim=1) == risks.argmin(dim=1)).float()
        loss_dict["diag_gate_entropy"] = gate_entropy.mean().detach()
        loss_dict["diag_oracle_match"] = oracle_match.mean().detach()
        loss_dict["diag_hard_fake_count"] = hard_fake_mask.float().sum().detach()
        loss_dict["diag_patch_entropy"] = entropy.mean().detach()
        loss_dict["diag_patch_variance"] = tile_variance.mean().detach()
        loss_dict["diag_mean_q_base"] = gate_probs[:, 0].mean().detach()
        loss_dict["diag_mean_q_artifact"] = gate_probs[:, 1].mean().detach()
        loss_dict["diag_mean_q_structure"] = gate_probs[:, 2].mean().detach()
        loss_dict["diag_mean_q_patch"] = gate_probs[:, 3].mean().detach()
        loss_dict["diag_family_delta_artifact"] = family_delta[:, 1].mean().detach()
        loss_dict["diag_family_delta_structure"] = family_delta[:, 2].mean().detach()
        loss_dict["diag_family_delta_patch"] = family_delta[:, 3].mean().detach()
        loss_dict["diag_family_margin_artifact"] = art_margin.mean().detach()
        loss_dict["diag_family_margin_structure"] = structure_margin.mean().detach()
        loss_dict["diag_family_margin_patch"] = patch_margin.mean().detach()
        loss_dict["diag_best_spec_correction_orig"] = correction_orig.max(dim=1).values.mean().detach()
        loss_dict["diag_best_spec_correction_family"] = correction_family.max(dim=1).values.mean().detach()

        if self.args.cie_debug_log:
            print(
                "[CIE_IAPL] hard_fake_count={:.0f} tile_entropy_mean={:.4f} tile_variance_mean={:.4f}".format(
                    loss_dict["diag_hard_fake_count"].item(),
                    loss_dict["diag_patch_entropy"].item(),
                    loss_dict["diag_patch_variance"].item(),
                )
            )

        return loss_dict
