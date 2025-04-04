# Background-randomized Post-Hoc Concept Embeddings

This is the code for our paper *On Background Bias of Post-Hoc Concept Embeddings in Computer Vision DNNs*.
The code is loosely building upon that of the paper on [local concept embeddings](http://arxiv.org/abs/2311.14435) published [here](https://github.com/continental/local-concept-embeddings).

## Overview
The code in the library `bg_randomized_loce` features functionality to
- load typical concept segmentation datasets
  (`data_structures` submodule, and script `scripts/voc2coco.py` to transform Pascal VOC2012 into MS COCO data format)
- creating synthetic backgrounds (`synthetic_bg_generation` submodule and `scripts/generate_backgrounds.py` script; requires the Würstchen model installed),
- pasting concept foregrounds from a concept segmentation dataset onto randomized backgrounds from any background dataset (`background_pasting` submodule);  
  this includes:
  - plain backgrounds (supported: Places205, backgrounds generated using `synthetic_bg_generation`)
  - merging different backgrounds into a Voronoi-diagram shaped puzzle pattern
- capture activation maps of common models (`hooks` submodule),
- train and test local and global concept embeddings on these activations (`loce` submodule),
- and some `utils` for handling the LoCE pickle storage format and for evaluation.
- Experiments conducted in the paper can be found under `./experiments`, and evaluation notebooks under `./notebooks`.

Please check the paper for details. (TODO: link)

### Installation
For installing the main library dependencies, install the `poetry` package manager and call in the project root:
```bash
poetry install
```

**Würstchen** diffuser uses different venv with exact requirements in 
```
./background_randomization/diffuser_exact_requirements.txt
```

## Paper Ref
If you find the code useful, please cite the following paper
```bibtex
@inproceedings{schwalbe2025background,
  title        = {On Background Bias of Post-Hoc Concept Embeddings in Computer Vision DNNs},
  author       = {Gesina Schwalbe and
                  Georgii Mikriukov and
                  Stavros Gerolymatos and 
                  Edgar Heinert and
                  Annika M{\"{u}}tze and
                  Mert Keser and
                  Alois Knoll and
                  Matthias Rottmann},
  booktitle    = {Explainable Artificial Intelligence - 3rd World Conference, xAI
                  2024, },
  series       = {Communications in Computer and Information Science},
  publisher    = {Springer},
  year         = {2025},
}
```