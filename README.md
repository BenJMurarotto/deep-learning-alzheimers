# deep-learning-alzheimers
by Ben Murarotto, Benson Kachappilly & Andrew Ly.

## COSC591 UNE Capstone Project
The University of New England (UNE) plus the wider academic and scientific communities are committed to pushing the boundaries of technology to improve society, particularly in the field of health.  Alzheimer’s Disease (AD) affects roughly 16 in every 1000 Australians and is the cause of approximately 9.5% of Australian deaths (AIHW https://www.aihw.gov.au/reports/dementia/dementia-in-aus/contents/how-many-people-have-dementia/prevalence-of-dementia). The diagnosis of AD is a challenge that was met by Hajati et al. 2025, who demonstrated that KAN deep learning models could be utilised to detect AD in retinal images and highlighted the applications of AI in medical diagnostics. Building on this foundation, this project delivers a software tool which gives end users access to powerful trained computer vision models that aid in diagnosing AD. The previous UNE student group had explored the potential of Vision Transformers (ViTs) for this task, our team directly implemented and deployed pretrained ViT models. This allowed us to evaluate their diagnostic performance in practice and provide a more accessible, end-to-end application for AD detection using retinal imaging. 

This was achieved by: 
    Designing preprocessing functions to vectorise 3D oct image files. 
    Training and optimising transformer models. 
    Creating a locally hosted dashboard to access model weights. 

This repo is dual purposed:
    1. It contains the training pipelines used for our vision transformer models.
    2. It contains code for running the streamlit dashboard to input OCT slices to the model.

The notebook folder contains Jupyter notebooks for each model iteration trained.
**To replicate the testing environment:**  Please install the dependencies found in the requirements.txt file of the root directory. You will need to extract the Oregon University Health and Science dataset into the folder data/ 
By executing the cells in the corresponding notebook you can create the model we developped or tweak it to suit your needs.
In the case that you are unable to train a model on your system, we will make the models publically available to download.

**To access the dashboard:** Please install the dependencies in the requirements.txt file located in the folder called frontend/. You will also need to ensure the exported model folder containing the model and its weights is in this same directory. As mentioned above you can either use the notebook to replicate a model OR download it online.
To locally host a streamlit dashboard on your system you will need to run the command streamlit run app_streamlit.py in the console.
