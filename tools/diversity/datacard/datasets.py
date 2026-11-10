"""Dataset specs: which motion files make up a dataset version, grouped by source group / category.

A spec is a JSON file (or dict)::

    {
      "name": "v7bones_sample30",
      "robot": "rp1_28dof",                      # optional, inferred from manifest qpos_dim
      "description": "mimic_v7 30% clip sample",
      "groups": [
        {"group": "md", "dir": "/data/mimic_v7/md"},                        # any4hdmi dir (manifest.json + motions/)
        {"group": "md", "motions": "/tmp/train/md",                         # flat copy of motions/*.npz ...
         "manifest": "/tmp/train_manifests/md/manifest.json"}               # ... with the manifest kept elsewhere
      ],
      "weights": {"md": 0.14, "getup": 0.30},     # optional sampling weights of the training mixture
      "subsets": {"getup": "/data/mimic_v7/getup/manifest.json"}           # re-weighted subsets (manifest or name list)
    }

Several entries may share a group (e.g. a sample plus extra files); entries are de-duplicated on (group, clip name).
For target sets the group is the evaluation category (locomotion / manipulation / ground / dance).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .motion_io import MOTION_EXTS, clip_names, source_id, strip_motion_ext  # noqa: F401  (re-exported)

DEFAULT_RECORDING_PATTERNS = (r"__\d+_\d+$", r"_M$")
clip_source_id = source_id


def recording_id(source: str, patterns: tuple[str, ...] | None = None) -> str:
    """Coarser id for leakage control: strips frame-range chunk suffixes (``__000000_000256``) and mirror tags (``_M``)."""
    pats = DEFAULT_RECORDING_PATTERNS if patterns is None else patterns
    out = source
    for p in pats:
        out = re.sub(p, "", out)
    return out


@dataclass
class ClipRef:
    path: str
    name: str
    source: str
    group: str
    timestep: float
    qpos_dim: int | None = None


@dataclass
class GroupSpec:
    group: str
    dir: str | None = None
    motions: str | None = None
    manifest: str | None = None
    files: list[str] | None = None
    timestep: float | None = None


@dataclass
class DatasetSpec:
    name: str
    groups: list[GroupSpec]
    robot: str | None = None
    description: str = ""
    weights: dict[str, float] = field(default_factory=dict)
    subsets: dict[str, str | list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSpec":
        groups = [GroupSpec(**g) for g in d["groups"]]
        return cls(name=d["name"], groups=groups, robot=d.get("robot"), description=d.get("description", ""),
                   weights=dict(d.get("weights") or {}), subsets=dict(d.get("subsets") or {}))

    @classmethod
    def load(cls, path: str | Path) -> "DatasetSpec":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)


def _group_entries(g: GroupSpec) -> list[ClipRef]:
    manifest_path = Path(g.manifest) if g.manifest else (Path(g.dir) / "manifest.json" if g.dir else None)
    manifest = {}
    if manifest_path is not None and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    names = clip_names(manifest) if manifest else None
    timestep = float(g.timestep or manifest.get("timestep", 0.02))
    qdim = manifest.get("qpos_dim")
    if g.files is not None:
        files = [Path(f) for f in g.files]
    else:
        if g.motions is not None:
            motions = Path(g.motions)
        elif g.dir is not None:
            motions = Path(g.dir) / manifest.get("motions_subdir", "motions")
        else:
            raise ValueError(f"group {g.group}: need 'dir', 'motions' or 'files'")
        files = sorted(motions.glob("*.npz"))
    out = []
    for f in files:
        idx = int(f.stem) if f.stem.isdigit() else None
        raw = names[idx] if names is not None and idx is not None and idx < len(names) else f.stem
        name = strip_motion_ext(raw)
        out.append(ClipRef(str(f), name, clip_source_id(raw), g.group, timestep, qdim))
    return out


def resolve_entries(spec: DatasetSpec) -> list[ClipRef]:
    seen, out = set(), []
    for g in spec.groups:
        for e in _group_entries(g):
            key = (e.group, e.name)
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
    return out


def subset_names(spec: DatasetSpec) -> dict[str, list[str]]:
    """Clip names (extension stripped) of each re-weighted subset in ``spec.subsets``."""
    out = {}
    for k, v in spec.subsets.items():
        if isinstance(v, str):
            with open(v) as f:
                names = clip_names(json.load(f)) or []
        else:
            names = list(v)
        out[k] = [strip_motion_ext(n) for n in names]
    return out


def spec_from_dirs(name: str, dirs: list[str], robot: str | None = None) -> DatasetSpec:
    """``dirs`` items are ``PATH`` or ``GROUP=PATH`` (any4hdmi layout); group defaults to the directory name."""
    groups = []
    for item in dirs:
        group, _, path = item.rpartition("=") if "=" in item else ("", "", item)
        groups.append(GroupSpec(group=group or Path(path).name, dir=path))
    return DatasetSpec(name=name, groups=groups, robot=robot)
