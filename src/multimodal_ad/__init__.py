"""multimodal_ad: package foundation for the multimodal AD detection project.

Scientific code from the five legacy Colab notebooks (preserved
byte-for-byte under notebooks/legacy/) has been ported into `data` (typed
manifest, labeling, OASIS-3 adapter, volume preprocessing, augmentation,
splitting) and `models` (3D CNN, training, evaluation, Grad-CAM, AAL2
region ranking). `notebooks/*.ipynb` are small, newcomer-oriented notebooks
that call into this package on synthetic data. See
docs/legacy-notebooks-inventory.md for the original notebooks' behavior
audit and open ambiguities.
"""

__version__ = "0.1.0"
