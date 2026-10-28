import copy
from pathlib import Path

import yaml


class Cfg(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    def to_dict(self):
        return _unwrap(self)


def _wrap(x):
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _unwrap(x):
    if isinstance(x, dict):
        return {k: _unwrap(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_unwrap(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path):
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("inherit", None)
    if parent:
        raw = _merge(_load_raw(path.parent / parent), raw)
    return raw


def _set_dotted(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path, overrides=()):
    raw = _load_raw(path)
    for item in overrides:
        key, _, val = item.partition("=")
        _set_dotted(raw, key.strip(), yaml.safe_load(val))
    cfg = _wrap(raw)
    cfg.config_path = str(Path(path).resolve())
    return cfg
