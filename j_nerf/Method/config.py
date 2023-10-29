import yaml

from j_nerf.Data.config import Config

_cfg = Config()


def init_cfg(filename):
    print("Loading config from: ", filename)
    _cfg.load_from_file(filename)


def get_cfg():
    return _cfg


def update_cfg(**kwargs):
    _cfg.update(kwargs)


def save_cfg(save_file):
    with open(save_file, "w") as f:
        f.write(yaml.dump(_cfg.dump()))


def print_cfg():
    data = yaml.dump(_cfg.dump())
    # TODO: data keys are not sorted
    print(data)
