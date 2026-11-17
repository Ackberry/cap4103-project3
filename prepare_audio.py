# Extract the seeded 20-speaker subset and cache its MFCC features.
import argparse
from pathlib import Path
import zipfile
import numpy as np
from biometrics import voice_features


def main():
    parser = argparse.ArgumentParser(description='Extract the seeded 20-speaker subset and cache its MFCC features.')
    parser.add_argument('--archive', type=Path, default=Path(__file__).parent / 'AudioMNIST.zip')
    args = parser.parse_args()
    root = Path(__file__).parent / 'data'
    root.mkdir(exist_ok=True)
    speakers = np.random.default_rng(42).choice(np.arange(1, 61), 20, replace=False)
    selected = {f'{speaker:02d}' for speaker in speakers}
    with zipfile.ZipFile(args.archive) as archive:
        entries = sorted(n for n in archive.namelist()
                         if n.endswith('.wav') and Path(n).parent.name in selected)
        if len(entries) != 10000:
            raise ValueError(f'Expected 10,000 recordings for 20 speakers, found {len(entries)}.')
        features, labels, paths = [], [], []
        for i, name in enumerate(entries):
            relative = Path(Path(name).parent.name) / Path(name).name
            destination = root / 'audio' / relative
            destination.parent.mkdir(exist_ok=True, parents=True)
            if not destination.exists():
                destination.write_bytes(archive.read(name))
            features.append(voice_features(destination))
            labels.append(int(relative.parent.name))
            paths.append(str(relative))
            if (i + 1) % 500 == 0:
                print(f'Extracted {i+1}/{len(entries)} recordings', flush=True)
    np.savez_compressed(root / 'voice.npz', features=features, labels=labels, paths=paths)


if __name__ == '__main__':
    main()
