from pathlib import Path

from vfbow import Vocabulary

def convert(src: Path, dst: Path):
    with open(src, 'rb') as f:
        voc = Vocabulary.read_from(f)

    with open(dst, 'wb') as f:
        voc.write_to(f)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Convert vocabularies from fbow to vfbow format",
    )
    parser.add_argument("input", type=Path, help="Input (fbow) file")
    parser.add_argument("output", type=Path, nargs='?', help="Output (vfbow) file")

    args = parser.parse_args()
    src: Path = args.input
    dst: Path | None = args.output 
    if not dst:
        dst = src.with_suffix(".vfbow")
    
    convert(src, dst)


if __name__ == '__main__':
    main()