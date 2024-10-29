from pathlib import Path
import time
import cv2

from vfbow import Vocabulary

def load_features(paths: list[Path], descriptor: str):
    # Select detector
    match descriptor:
        case "orb":
            fdetector = cv2.ORB.create(2000)
        case "brisk":
            fdetector = cv2.BRISK.create()
        case "akaze":
            fdetector = cv2.AKAZE.create(cv2.AKAZE_DESCRIPTOR_MLDB, 0, 3, 1e-4)
        case "surf":
            # TODO
            fdetector = cv2.SURF.create(15, 4, 2)
        case _:
            raise ValueError(f'Invalid descriptor {descriptor}')
    
    print("Extracting features...")
    features = list()
    for path in paths:
        print(f"Reading image {path}")
        try:
            image = cv2.imread(str(path), 0)
        except:
            print("\tCould not open image")
            continue
        print("\tExtractign features...")
        kps, des = fdetector.detectAndCompute(image, None)
        features.append(des)
        print("\tDone")
    return features

def convert(src: Path, dst: Path):
    with open(src, 'rb') as f:
        voc = Vocabulary.read_from(f)

    with open(dst, 'wb') as f:
        voc.write_to(f)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Detect and transform features",
    )
    parser.add_argument("vocabulary", type=Path, help="Input (vfbow) file")
    parser.add_argument("images", type=Path, nargs='+', help="Path to images")
    parser.add_argument("--descriptor", '-d', choices=["orb", "brisk", "akaze", "surf"], required=False)

    args = parser.parse_args()
    src: Path = args.vocabulary
    imgs: list[Path] = args.images
    descriptor: str | None = parser.descriptor

    with open(src, 'rb') as f:
        voc = Vocabulary.read_from(f)
    
    print(f"Vocabulary descriptor name={voc.desc_name}")
    
    if descriptor is None:
        descriptor = voc.desc_name
    
    features = load_features(imgs, descriptor)
    print(f"{sum(feature.shape[0] for feature in features)}x{features[0].shape[1]} features")

    t_start = time.time_ns()
    vv, _ = voc.transform(features)
    t_end = time.time_ns()
    print(f"Took {(t_end-t_start/1e6)}ms")

    # cout<<vv.begin()->first<<" "<<vv.begin()->second<<endl;
    # cout<<vv.rbegin()->first<<" "<<vv.rbegin()->second<<endl;
    # for(auto v:vv)
    #     cout<<v.first<<" ";
    # cout<<endl;

if __name__ == '__main__':
    main()