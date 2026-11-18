"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q
"""
from collections import defaultdict

import yaml

from ..common import ERROR, parse_yolo_line


DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_CODES = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def _clip(text):
    """Cắt box về [0, 1]. Trả dòng mới, hoặc None nếu sau cắt không còn diện tích."""
    cls, cx, cy, w, h = parse_yolo_line(text)
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    nw, nh = x2 - x1, y2 - y1
    if nw <= 0 or nh <= 0:
        return None
    return f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {nw:.6f} {nh:.6f}"


def _row(image, line, action, before, after):
    return {"image": image, "line": line if line is not None else "", "action": action, "before": before, "after": after}


def _remove_image(ds, stem):
    image = ds.image_path(stem)
    if image is not None and image.exists():
        image.unlink()
    label = ds.label_path(stem)
    if label.exists():
        label.unlink()


def _rename_classes(ds, cfg):
    data = {}
    if ds.yaml_path.exists():
        data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
    before = data.get("names")
    data["names"] = list(cfg["classes"])
    ds.yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return before


def fix(ds, issues, cfg):
    errors = [i for i in issues if i.severity == ERROR]
    fixes = []
    removed = set()
    for issue in errors:
        if issue.image in removed:
            continue
        if issue.code.startswith("DUP_IMAGE_") or issue.code in REMOVE_CODES:
            _remove_image(ds, issue.image)
            removed.add(issue.image)
            fixes.append(_row(issue.image, issue.line, "remove_image", issue.image, "đã xoá ảnh và nhãn"))

    renamed = False
    for issue in errors:
        if issue.code != "BAD_CLASS_NAME" or renamed:
            continue
        before = _rename_classes(ds, cfg)
        fixes.append(_row(issue.image, issue.line, "rename_classes", str(before), str(list(cfg["classes"]))))
        renamed = True

    for issue in errors:
        if issue.code == "MISSING_LABEL_FILE":
            fixes.append(_row(issue.image, issue.line, "needs_label", "(không có file)", "chuyển người gán"))

    by_image = defaultdict(list)
    for issue in errors:
        if issue.image in removed or issue.line is None:
            continue
        if issue.code in DROP_CODES or issue.code == "BOX_OUT_OF_IMAGE":
            by_image[issue.image].append(issue)

    for stem, group in by_image.items():
        path = ds.label_path(stem)
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        drop_at = set()
        clip_at = set()
        for issue in group:
            if issue.line < 1 or issue.line > len(lines):
                continue
            if issue.code in DROP_CODES:
                drop_at.add(issue.line)
            elif issue.code == "BOX_OUT_OF_IMAGE":
                clip_at.add(issue.line)
        clip_at -= drop_at
        kept = []
        for number, text in enumerate(lines, 1):
            if number in drop_at:
                fixes.append(_row(stem, number, "drop_line", text, ""))
                continue
            if number in clip_at:
                clipped = _clip(text)
                if clipped is None:
                    fixes.append(_row(stem, number, "drop_line", text, ""))
                    continue
                fixes.append(_row(stem, number, "clip_box", text, clipped))
                kept.append(clipped)
                continue
            kept.append(text)
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return fixes
