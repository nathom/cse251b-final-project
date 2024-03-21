import re
import sys

import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

path = sys.argv[1]


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
        avgList = []
        maxList = []

        # from percent1 values
        f2048 = {}
        for i in range(16):
            f2048[2**i] = [0] * len(info)

        # from percent2 values
        s2048 = {}
        for i in range(16):
            s2048[2**i] = [0] * len(info)

        for i, (avg, max, tupleList) in enumerate(info):
            avgList.append(avg)
            maxList.append(max)
            for tile, percent1, percent2 in tupleList:
                f2048[tile][i] = percent1
                s2048[tile][i] = percent2

    if sys.argv[2] == "am":
        window_size = 10
        max_list_smooth = uniform_filter1d(maxList, size=window_size)
        expertLevelHuman = [100000] * len(info)
        plt.plot(avgList, label="Average")
        plt.plot(max_list_smooth, label="Max (moving average)")
        plt.plot(expertLevelHuman, linestyle="dashed")
        plt.text(2000, 105000, "Expert Level Human")

        plt.xlabel("Thousands of Games")
        plt.ylabel("Score")
        plt.title("Games versus Score")
        plt.grid(True)
        plt.legend()
        plt.show()

    # first percent
    if sys.argv[2] == "f":
        plt.plot(f2048[2048], label="2048")
        plt.plot(f2048[4096], label="4096")
        plt.plot(f2048[8192], label="8192")
        plt.plot(f2048[16384], label="16384")

        plt.xlabel("Thousands of Games")
        plt.ylabel("Tile Achieved")
        plt.title("Games versus Tile Achieved")
        plt.grid(True)
        plt.legend()
        plt.show()

    # second percent
    if sys.argv[2] == "s":
        plt.plot(s2048[2048], label="2048")
        plt.plot(s2048[4096], label="4096")
        plt.plot(s2048[8192], label="8192")
        plt.plot(s2048[16384], label="16384")

        plt.xlabel("Thousands of Games")
        plt.ylabel("Percent Games with Losing Tile")
        plt.title("Games versus Losing Tile")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1])
