from jittor import nn


def build_from_cfg(cfg, registry, **kwargs):
    if isinstance(cfg, str):
        return registry.get(cfg)(**kwargs)
    elif isinstance(cfg, dict):
        args = cfg.copy()
        args.update(kwargs)
        obj_type = args.pop("type")
        obj_cls = registry.get(obj_type)
        try:
            module = obj_cls(**args)
        except TypeError as e:
            if "<class" not in str(e):
                e = f"{obj_cls}.{e}"
            raise TypeError(e)

        return module
    elif isinstance(cfg, list):
        return nn.Sequential([build_from_cfg(c, registry, **kwargs) for c in cfg])
    elif cfg is None:
        return None
    else:
        raise TypeError(f"type {type(cfg)} not support")
