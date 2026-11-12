#!/usr/bin/env python3
"""Install optional short command symlinks for an exported Skill collection."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, default=Path.home() / '.local/bin')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commands = json.loads((root / 'commands.json').read_text())
    plans = []
    for name, definition in commands.items():
        target = root / 'skills' / definition['skill'] / definition['entry']
        link = args.bin_dir.expanduser().resolve() / name
        if not target.is_file() or not target.resolve().is_relative_to(root):
            raise SystemExit('Missing or invalid command entry: ' + name)
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise SystemExit('Existing command will not be overwritten: ' + str(link))
        plans.append((target, link))
    for target, link in plans:
        link.parent.mkdir(parents=True, exist_ok=True)
        target.chmod(target.stat().st_mode | 0o111)
        link.symlink_to(target)
        print(link)


if __name__ == '__main__':
    main()
