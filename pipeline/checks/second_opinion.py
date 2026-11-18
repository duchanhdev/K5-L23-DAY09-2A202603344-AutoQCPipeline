"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds


def _issue(stem, code, message, line=None, extra=None):
    return Issue("second_opinion", WARN, code, stem, message, line, extra or {})


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")
    so = cfg["second_opinion"]
    conf_min = so["conf"]
    min_area = so["min_area_px"]
    fp_conf = so["fp_conf"]
    iou_missing = so["iou_missing"]
    iou_match = so["iou_match"]
    iou_loose = so["iou_loose"]
    n_cls = len(ds.classes)
    issues = []
    for image in ds.image_files():
        stem = image.stem
        if stem not in preds or ds.read_label_lines(stem) is None:
            continue
        size = ds.image_size(stem)
        if not size:
            continue
        width, height = size
        labels = []
        for line_no, text in ds.read_label_lines(stem):
            try:
                parsed = parse_yolo_line(text)
            except ValueError:
                continue
            if 0 <= parsed[0] < n_cls:
                labels.append((line_no, parsed))
        for pred in preds[stem]:
            cls, pconf = int(pred[0]), float(pred[1])
            box = tuple(pred[2:6])
            if pconf < conf_min or box[2] * width * box[3] * height < min_area:
                continue
            cands = []
            for line_no, lab in labels:
                iou = iou_xywh(box, lab[1:])
                if iou >= iou_missing:
                    cands.append((iou, lab[0], line_no))
            same = [c for c in cands if c[1] == cls]
            other = [c for c in cands if c[1] != cls]
            if not cands:
                issues.append(_issue(stem, "POSSIBLE_MISSING", "model thấy vật, nhãn không có box", extra={"conf": pconf}))
            elif same:
                best = max(c[0] for c in same)
                if best < iou_loose:
                    issues.append(_issue(stem, "POSSIBLE_LOOSE_BOX", "đúng lớp nhưng box lệch", extra={"iou": round(best, 4), "conf": pconf}))
            else:
                best = max(c[0] for c in other)
                if best >= iou_match:
                    issues.append(_issue(stem, "POSSIBLE_WRONG_CLASS", "cùng vật nhưng khác lớp", extra={"iou": round(best, 4), "conf": pconf}))
        for line_no, lab in labels:
            best = 0.0
            covered = False
            for pred in preds[stem]:
                if float(pred[1]) < fp_conf:
                    continue
                iou = iou_xywh(lab[1:], tuple(pred[2:6]))
                if iou > best:
                    best = iou
                if iou >= iou_missing:
                    covered = True
                    break
            if not covered:
                issues.append(_issue(stem, "POSSIBLE_FALSE_POS", "box nhãn không khớp dự đoán nào", line_no, {"iou": round(best, 4)}))
    return issues
