"""
Golden fixture generator for Arnold characterization tests.

Run ONCE (already run) to (re)create the JSON/NPZ golden files that the tests
compare against:

    cd /home/agent/work/Arnold && uv run python tests/golden/generate_goldens.py

The tests reconstruct the same seeded modules and assert their fresh outputs
match these stored goldens.  A refactor that silently changes behavior will
therefore make the tests fail.

Everything here is CPU-only and deterministic (torch.manual_seed).
"""

import os
import sys
import json

import torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
sys.path.insert(0, TESTS_DIR)

import helpers  # noqa: E402  (tests/helpers.py)


GOLDEN_JSON = os.path.join(HERE, "goldens.json")
GOLDEN_NPZ = os.path.join(HERE, "goldens.npz")


########################################
#           Module builders            #
########################################


def build_vocab():
    from arnold.torch_model.sensorimotor_vocabulary import (
        SensorimotorVocabulary,
    )

    torch.manual_seed(0)
    return SensorimotorVocabulary(embed_dim=helpers.EMBED_DIM)


def build_sensory_encoder():
    from arnold.torch_model.sensory_encoder import SensoryEncoder

    torch.manual_seed(1)
    return SensoryEncoder(
        history_len=helpers.HISTORY_LEN,
        embed_dim=helpers.EMBED_DIM,
    )


def build_mlp():
    from arnold.torch_model.mlp import MLP

    torch.manual_seed(2)
    return MLP(input_dim=8, hidden_dims=(16, 4), activation="silu")


def build_body_tokenizer(op):
    from arnold.torch_model.body_tokenizer import BodyTokenizer

    groups = op.get_body_groups("per_body")
    torch.manual_seed(3)
    tok = BodyTokenizer(
        groups={"legs": groups},
        history_len=helpers.HISTORY_LEN,
        embed_dim=helpers.EMBED_DIM,
    )
    return tok


def build_action_tokenizer(ap):
    from arnold.torch_model.action_tokenizer import ActionTokenizer

    grouping = ap.get_muscle_grouping("hybrid")
    torch.manual_seed(4)
    tok = ActionTokenizer(
        groupings={"legs": grouping},
        action_signatures={"legs": ap.action_signatures},
        embed_dim=helpers.EMBED_DIM,
        strategy="hybrid",
    )
    return tok, grouping


def build_normalizer():
    from arnold.torch_model.normalization import SignatureNormalizerModule

    return SignatureNormalizerModule()


########################################
#              Generation              #
########################################


def main():
    goldens = {}
    npz = {}

    op = helpers.make_observation_parser()
    ap = helpers.make_action_parser()

    # -- ObservationParser structural facts --
    goldens["obs_n_elements"] = op.n_obs_elements
    goldens["obs_n_specs"] = len(op.obs_specs)
    goldens["obs_n_signatures"] = len(op.obs_signatures)
    goldens["obs_spec_names_sizes"] = [(s.name, s.size) for s in op.obs_specs]
    goldens["obs_groups_scalar"] = len(op.get_body_groups("scalar"))
    goldens["obs_groups_per_spec"] = len(op.get_body_groups("per_spec"))
    goldens["obs_groups_per_body"] = len(op.get_body_groups("per_body"))

    # -- ActionParser groupings per strategy --
    ap_goldens = {}
    for strat in ("anatomical", "functional", "hybrid"):
        mg = ap.get_muscle_grouping(strat)
        ap_goldens[strat] = {
            "n_groups": mg.n_groups,
            "n_muscles": mg.n_muscles,
            "group_order": mg.group_order,
            "muscle_to_group": mg.muscle_to_group,
            "granule_base_map": mg.granule_base_map,
        }
    goldens["action_parser"] = ap_goldens

    # -- SensorimotorVocabulary --
    vocab = build_vocab()
    goldens["vocab_size"] = vocab.vocab_size
    sig_a = ("DELT1", "r", "muscle", "activation")
    sig_b = ("femur", "l", "position", "x")
    emb_a = vocab.get_embedding(sig_a)
    emb_b = vocab.get_embedding(sig_b)
    npz["vocab_emb_a"] = emb_a.detach().numpy()
    npz["vocab_emb_b"] = emb_b.detach().numpy()
    batch = vocab.get_embedding_batch([sig_a, sig_b, sig_a])
    npz["vocab_batch"] = batch.detach().numpy()

    # -- SensoryEncoder --
    enc = build_sensory_encoder()
    torch.manual_seed(100)
    x = torch.randn(2, 6, helpers.HISTORY_LEN)
    npz["sensory_out"] = enc(x).detach().numpy()

    # -- MLP --
    mlp = build_mlp()
    torch.manual_seed(101)
    xm = torch.randn(3, 8)
    npz["mlp_out"] = mlp(xm).detach().numpy()

    # -- BodyTokenizer --
    tok = build_body_tokenizer(op)
    goldens["body_tok_n_groups"] = len(tok.get_group_signatures("legs"))
    goldens["body_tok_types"] = tok._expert_type_orders["legs"]
    torch.manual_seed(102)
    obs_ts = torch.randn(2, op.n_obs_elements, helpers.HISTORY_LEN)
    body_out = tok.encode(obs_ts, "legs")
    goldens["body_tok_out_shape"] = list(body_out.shape)
    npz["body_tok_out"] = body_out.detach().numpy()

    # -- ActionTokenizer --
    atok, grouping = build_action_tokenizer(ap)
    goldens["atok_n_groups"] = grouping.n_groups
    goldens["atok_group_sizes"] = atok.expert_group_sizes["legs"]
    goldens["atok_group_signatures"] = [
        list(s) for s in atok.get_group_signatures("legs")
    ]
    goldens["atok_muscle_sigs"] = [
        list(s) for s in atok.get_muscle_signatures("legs")
    ]
    torch.manual_seed(103)
    group_out = torch.randn(2, grouping.n_groups, helpers.EMBED_DIM)
    mean, hidden = atok.decode(group_out, "legs")
    goldens["atok_mean_shape"] = list(mean.shape)
    goldens["atok_hidden_shape"] = list(hidden.shape)
    npz["atok_mean"] = mean.detach().numpy()
    npz["atok_hidden"] = hidden.detach().numpy()

    # -- SignatureNormalizerModule --
    norm = build_normalizer()
    sigs = [("a", "b"), ("c", "d"), ("e", "f")]
    torch.manual_seed(104)
    vals1 = torch.randn(4, 3, helpers.HISTORY_LEN)
    vals2 = torch.randn(4, 3, helpers.HISTORY_LEN)
    norm.update(sigs, vals1)
    norm.update(sigs, vals2)
    # capture stats
    norm_stats = {}
    for k, (count, mean_t, m2_t) in norm.stats.items():
        norm_stats[k] = {
            "count": int(count),
            "mean": float(mean_t),
            "M2": float(m2_t),
        }
    goldens["norm_stats"] = norm_stats
    test_vals = torch.arange(
        3 * helpers.HISTORY_LEN,
        dtype=torch.float32,
    ).reshape(1, 3, helpers.HISTORY_LEN)
    npz["norm_normalized"] = norm.normalize(sigs, test_vals).detach().numpy()

    # -- policy_moe deterministic helpers --
    # Build a tiny MoE policy purely to call its pure helper methods.
    from arnold.torch_model.policy_moe import MoELatticePolicy

    torch.manual_seed(5)
    action_dim = 6
    latent_dim = 4
    sparse = MoELatticePolicy(
        state_dim=8,
        action_dim=action_dim,
        shared_units=(),
        shared_expert_units=(latent_dim,),
        num_experts=4,
        top_k=2,
        expert_units=(latent_dim,),
        gate_units=(8,),
        value_units=(8,),
    )
    torch.manual_seed(106)
    top_k_indices = torch.tensor([[0, 1], [2, 3], [0, 2]], dtype=torch.long)
    gate_probs = torch.softmax(torch.randn(3, 4), dim=-1)
    lb_sparse, lb_sparse_stats = sparse.compute_sparse_balance_loss(
        top_k_indices,
        gate_probs,
    )
    goldens["moe_sparse_lb"] = float(lb_sparse.item())

    torch.manual_seed(6)
    soft = MoELatticePolicy(
        state_dim=8,
        action_dim=action_dim,
        shared_units=(),
        shared_expert_units=(latent_dim,),
        num_experts=4,
        top_k=4,
        expert_units=(latent_dim,),
        gate_units=(8,),
        value_units=(8,),
    )
    lb_soft, lb_soft_stats = soft.compute_soft_balance_loss(gate_probs)
    goldens["moe_soft_lb"] = float(lb_soft.item())
    goldens["moe_soft_stats"] = {k: float(v) for k, v in lb_soft_stats.items()}

    # _compute_cov_factor: shared-only path (default cov_from_experts=False)
    top_k_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]])
    cov_f, diag_std, latent_std = sparse._compute_cov_factor(
        top_k_indices,
        top_k_weights,
    )
    goldens["moe_cov_shared_shape"] = list(cov_f.shape)
    npz["moe_cov_shared"] = cov_f.detach().numpy()
    npz["moe_diag_std"] = diag_std.detach().numpy()
    npz["moe_latent_std"] = latent_std.detach().numpy()

    # cov_from_experts=True path (per-sample)
    torch.manual_seed(7)
    per = MoELatticePolicy(
        state_dim=8,
        action_dim=action_dim,
        shared_units=(),
        shared_expert_units=(latent_dim,),
        num_experts=4,
        top_k=2,
        expert_units=(latent_dim,),
        gate_units=(8,),
        value_units=(8,),
        cov_from_experts=True,
    )
    cov_pf, _, _ = per._compute_cov_factor(top_k_indices, top_k_weights)
    goldens["moe_cov_persample_shape"] = list(cov_pf.shape)
    npz["moe_cov_persample"] = cov_pf.detach().numpy()

    # -- Policy forward passes (lattice / moe / transformer / gate_moe) --
    import tempfile

    op2, ap2, obs = helpers.make_policy_io()
    state_dim = op2.n_obs_elements
    action_dim = len(ap2.action_signatures)
    obs_sigs = op2.obs_signatures
    act_sigs = ap2.action_signatures

    def _dump(prefix, out):
        names = ["mean", "cov", "diag_std", "value", "latent_std", "lb"]
        for name, tensor in zip(names, out, strict=False):
            if tensor is None:
                goldens[f"{prefix}_{name}"] = None
            else:
                npz[f"{prefix}_{name}"] = tensor.detach().numpy()
                goldens[f"{prefix}_{name}_shape"] = list(tensor.shape)

    lat = helpers.build_lattice(state_dim, action_dim)
    _dump("lat", lat(obs, obs_sigs, act_sigs, "legs"))

    moe = helpers.build_moe(state_dim, action_dim)
    _dump("moe_fwd", moe(obs, obs_sigs, act_sigs, "legs"))

    tp = helpers.build_transformer(op2)
    _dump("tp", tp(obs, obs_sigs, act_sigs, "legs"))

    tpg = helpers.build_transformer_granulated(op2, ap2)
    _dump("tpg", tpg(obs, obs_sigs, act_sigs, "legs"))

    tmp = tempfile.mkdtemp()
    gm = helpers.build_gate_moe(state_dim, action_dim, tmp)
    _dump("gm", gm(obs, obs_sigs, act_sigs, "legs"))

    # -- OBCMemory GAE --
    from arnold.memory import OBCMemory

    m = OBCMemory()
    torch.manual_seed(107)
    n = 5
    for i in range(n):
        m.states.append(torch.zeros(2, helpers.HISTORY_LEN))
        m.obs_signatures.append([("a", "b")])
        m.action_signatures.append([("c", "d")])
        m.policy_actions.append(torch.zeros(3))
        m.expert_actions.append(torch.zeros(3))
        m.values.append(torch.tensor([float(i) * 0.5]))
        m.log_probs.append(torch.zeros(1))
    m.rewards = [1.0, 0.5, -0.2, 0.8, 1.5]
    m.masks = [1.0, 1.0, 1.0, 1.0, 0.0]
    batch = m.to_batch(gamma=0.99, tau=0.95)
    npz["gae_returns"] = batch.returns.detach().numpy()
    npz["gae_advantages"] = batch.advantages.detach().numpy()

    # -- write --
    with open(GOLDEN_JSON, "w") as f:
        json.dump(goldens, f, indent=2, sort_keys=True)
    np.savez(GOLDEN_NPZ, **npz)
    print(f"wrote {GOLDEN_JSON}")
    print(f"wrote {GOLDEN_NPZ} with keys: {sorted(npz.keys())}")


if __name__ == "__main__":
    main()
