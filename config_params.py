import torch
from collections import defaultdict

def get_params_for_weight_decay_optimization(module, config):
    """
    Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and biases will have no weight decay but the rest will.
    """
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}
    blacklist_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    for module_ in module.modules():
        if isinstance(module_, blacklist_modules) or (
            config.weight_decay == 0.0
        ):  # also include all parameters here if no weight decay is being done
            no_weight_decay_params["params"].extend(
                [
                    p
                    for p in list(module_._parameters.values())
                    if (p is not None) and p.requires_grad
                ]
            )
        else:
            for n, p in list(module_._parameters.items()):
                if p is not None and p.requires_grad:
                    if n != "bias":
                        weight_decay_params["params"].append(p)
                    else:
                        no_weight_decay_params["params"].append(p)
    param_dict = {
        pn: p
        for pn, p in module.named_parameters()
        if p is not None and p.requires_grad
    }
    #print(len(no_weight_decay_params["params"]), "<--no weight", "weight--->", len( weight_decay_params["params"]), 'param_dict', len(param_dict.keys()))
    assert len(no_weight_decay_params["params"]) + len(
        weight_decay_params["params"]
    ) == len(
        param_dict.keys()
    ), "Number of params in both groups != total number of trainable params"
    if config.weight_decay == 0.0:
        # only return a single param group if no weight decay is being used anyway
        return [no_weight_decay_params]
    return [weight_decay_params, no_weight_decay_params]


def configure_param_groups(model, config):
    """
    Configures the different parameter groups in the model for training.
    If a separate learning rate for the image prefix is provided, we separate out the groups here.
    Additionally, parameters to which weight decay shouldn't be applied (layernorms / biases) are separated.
    """
    if config.image_enc_lr is not None:
        # get the params for the image prefix / proj
        image_enc_params = get_params_for_weight_decay_optimization(
            model.image_prefix.enc, config
        )
        for pdict in image_enc_params:
            pdict["lr"] = config.image_enc_lr
        image_proj_params = get_params_for_weight_decay_optimization(
            model.image_prefix.proj, config
        )
        # get the params for layernorm if it exists
        if config.use_image_embed_layernorm:
            image_ln_params = get_params_for_weight_decay_optimization(
                model.image_prefix.ln, config
            )
            image_proj_params += image_ln_params
        # get the params for the lm
        lm_params = get_params_for_weight_decay_optimization(model.transformer, config)
        lm_params +=get_params_for_weight_decay_optimization(model.lm_head, config)
        # get params for class head if it exists
        class_params = []
        if hasattr(model, "class_head") and model.class_head is not None:
            class_params = get_params_for_weight_decay_optimization(
                model.class_head, config
            )
        all_params = []
        for p in image_enc_params + lm_params + image_proj_params + class_params:
            if p["params"]:
                all_params.append(p)
    else:
        all_params = get_params_for_weight_decay_optimization(model, config)
    # merge param dicts with shared lr / wd values
    d = defaultdict(dict)
    for param_group in all_params:
        lr = param_group.get("lr", None)
        wd = param_group.get("weight_decay", None)
        key = f"lr_{lr}_wd_{wd}"
        if d[key].get("params") is None:
            d[key]["params"] = []
        d[key]["params"].extend(param_group["params"])
        if lr is not None:
            d[key]["lr"] = lr
        if wd is not None:
            d[key]["weight_decay"] = wd
    all_params = list(d.values())
    n_params = sum([len(d["params"]) for d in all_params])
    param_dict = {
        pn: p for pn, p in model.named_parameters() if p is not None and p.requires_grad
    }
    assert n_params == len(
        param_dict
    ), f"Some parameters are missing from param groups ({n_params} | {len(param_dict)})"
    # if we're using multiple param groups, set the min / max lr for each one[]
    # appropriately in deepspeed's scheduler
    config.deepspeed_config_params["scheduler"]["params"]["warmup_min_lr"] = [
        config.min_lr for _ in all_params
    ]
    config.deepspeed_config_params["scheduler"]["params"]["warmup_max_lr"] = [
        d.get("lr", config.lr) for d in all_params
    ]
    return all_params