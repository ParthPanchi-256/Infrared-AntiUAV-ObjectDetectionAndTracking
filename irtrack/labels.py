"""Parsing for the dim-small aircraft dataset label files (data/data_label/dataN.txt)."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Sequence:
    name: str                 # folder name, e.g. "data10"
    header_name: str          # seq name in label header, e.g. "data1"
    num_frames: int           # number of BMP frames on disk
    num_objects: int
    # frame index -> list of (object_id, x, y)
    frames: dict = field(default_factory=dict)

    def image_path(self, root: Path, idx: int) -> Path:
        return root / self.name / self.name / f"{idx}.bmp"


def parse_label_file(path: Path, num_frames: int | None = None) -> Sequence:
    lines = path.read_text().splitlines()
    # header fields are tab separated, some files contain empty fields (extra tabs)
    hdr = [t for t in lines[0].split("\t") if t.strip()]
    _, header_name, n, n_obj = hdr[:4]
    seq = Sequence(path.stem, header_name, int(n), int(n_obj))
    for line in lines[1:]:
        if not line.startswith("frame:"):
            continue
        tok = [t.strip() for t in line.split("\t")]
        idx = int(tok[0].split(":")[1])
        objs = []
        count = int(tok[1]) if len(tok) > 1 and tok[1] else 0
        for k in range(count):
            oid, x, y = tok[2 + 3 * k: 5 + 3 * k]
            objs.append((int(oid.split(":")[1]), int(x), int(y)))
        seq.frames[idx] = objs
    if num_frames is not None:
        # drop label lines that have no image (e.g. data1.txt frame:399)
        seq.frames = {i: o for i, o in seq.frames.items() if i < num_frames}
        seq.num_frames = num_frames
    return seq


def load_all(data_root: Path) -> list[Sequence]:
    seqs = []
    for lbl in sorted((data_root / "data_label").glob("data*.txt"), key=lambda p: int(p.stem[4:])):
        n = len(list((data_root / lbl.stem / lbl.stem).glob("*.bmp")))
        seqs.append(parse_label_file(lbl, n))
    return seqs
