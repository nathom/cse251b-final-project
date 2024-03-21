import re
import sys

path = sys.argv[1]


def paragraphs(s):
    return s.split("\n\n")


def parse_paragraph(p):
    lines = p.split("\n")
    m = re.match(r"\d+\s+avg\s+=\s+(\d+\.?\d*)\s+max\s+=\s+(\d+)", lines[0])
    assert m is not None
    avg, max = float(m.group(1)), int(m.group(2))
    # 256     100%    (0.3%)
    items = []
    for line in lines[1:]:
        if not line:
            continue
        print(f"'{line}'")
        m = re.match(r"\s+(\d+)\s+(\d+\.?\d*)%\s+\((\d+\.?\d*)\%\)\s*", line)
        assert m is not None
        tilen, perc, perc2 = int(m.group(1)), float(m.group(2)), float(m.group(3))
        items.append((tilen, perc, perc2))
    return avg, max, items


with open(path) as f:
    s = f.read()
    info = [parse_paragraph(p) for p in paragraphs(s)]
    print(info)
