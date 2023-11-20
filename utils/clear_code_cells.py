import os
import argparse
import json
from pathlib import Path


def clear_code_cells(nb_path):
    nb_path = Path(nb_path)
    nb_json = json.load(nb_path.open())
    for cell in nb_json['cells']:
        if cell['cell_type'] != 'code':
            continue

        cell['source'] = [
            "# ... WRITE YOUR CODE HERE ... #"
        ]

    new_nb_path = nb_path.parent / f'{nb_path.stem}_nocode.ipynb'
    json.dump(nb_json, new_nb_path.open('w'), indent=' ',  ensure_ascii=False)

    os.system(
        f'diff -u "{new_nb_path}" "{nb_path}" > '
        f'"{nb_path.parent / nb_path.stem}.patch"'
    )
    os.system(
        f'mv "{new_nb_path}" "{nb_path}"'
    )


def undo_clear_code_cells(nb_path):
    nb_path = Path(nb_path)
    patch_path = nb_path.parent / f'{nb_path.stem}.patch'
    os.system(
        f'patch "{nb_path}" "{patch_path}"'
    )
    os.system(
        f'rm "{patch_path}"'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'notebook',
        help='Path to the notebook from which to clear the code cells.'
    )
    parser.add_argument(
        '--undo',
        action='store_true',
        help='Add this flag to add the code back in the cells'
    )
    args = parser.parse_args()

    if args.undo:
        undo_clear_code_cells(args.notebook)
    else:
        clear_code_cells(args.notebook)
