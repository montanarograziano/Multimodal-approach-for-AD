# Multimodal approach for Dementia Detection
 
This repository contains notebooks for the experiments conducted for testing a multimodal approach for assessing Dementia from PET and MRI. The notebooks are separated in:
- *Dataseet_MRI.ipynb*, containing the code used to process the raw MRI data to create the final dataset.
- *Dataseet_PET.ipynb*, containing the code used to process the raw PET data to create the final dataset.
- *Training.ipynb*, which contains the code used for training and testing the various models.
- *Heatmaps.ipynb*, which contains the code used for generating heatmaps based on the GradCAM algorithm, using the plugin [*tf-keras-vis*](https://github.com/keisen/tf-keras-vis).
- *Exploration.ipynb*, which contains the code used for the evaluation of the heatmaps in order to rank the brain's zones.
