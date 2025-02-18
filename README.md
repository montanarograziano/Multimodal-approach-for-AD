*UPDATE*:
- **03/2024**: Published on Scientific Reports! Check the paper [here](https://www.nature.com/articles/s41598-024-56001-9).

# Multimodal approach for AD Detection
 
This repository contains notebooks for the experiments conducted for testing a multimodal approach for assessing AD from PET and MRI. The notebooks are separated in:
- *Dataset_MRI.ipynb*, containing the code used to process the raw MRI data to create the final dataset.
- *Dataset_PET.ipynb*, containing the code used to process the raw PET data to create the final dataset.
- *Training.ipynb*, which contains the code used for training and testing the various models.
- *Heatmaps.ipynb*, which contains the code used for generating heatmaps based on the GradCAM algorithm, using the plugin [*tf-keras-vis*](https://github.com/keisen/tf-keras-vis).
- *Exploration.ipynb*, which contains the code used for the evaluation of the heatmaps in order to rank the brain's zones.
- *images/3D Brain Plot.ipynb*, which contains the code used to generate the figures available in the paper.
