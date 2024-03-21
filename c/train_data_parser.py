import re
import sys


def paragraphs(s):
    return s.split("\n\n")


re_avg_max = re.compile(r"\d+\s+avg\s+=\s+(\d+\.?\d*)\s+max\s+=\s+(\d+)")
re_tile_info = re.compile(r"\s+(\d+)\s+(\d+\.?\d*)%\s+\((\d+\.?\d*)\%\)\s*")


def parse_paragraph(p):
    lines = p.split("\n")
    m = re_avg_max.match(lines[0])
    assert m is not None
    avg, max = float(m.group(1)), int(m.group(2))
    # 256     100%    (0.3%)
    items = []
    for line in lines[1:]:
        if not line:
            continue
        m = re_tile_info.match(line)
        assert m is not None
        tilen, perc, perc2 = int(m.group(1)), float(m.group(2)), float(m.group(3))
        items.append((tilen, perc, perc2))
    return avg, max, items


def main(path):
    with open(path) as f:
        s = f.read()
        info = [parse_paragraph(p) for p in paragraphs(s)]
        print(info)


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Missing file path"
    main(sys.argv[1])
