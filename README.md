# MLEnd-Rice-Or-Chips-Classifier

### Introduction
This project utilizes the MLEnd Yummy Dataset to build a machine learning pipeline capable of predicting whether a given image of a dish contains rice or chips. The pipeline includes stages such as data preprocessing, feature extraction, model training, and evaluation, with categorization performed dynamically using keyword matching in the code.

### Dataset Description
The dataset comprises 3,250 images from the MLEnd Yummy Dataset. The images are initially uncategorized; categorization into two classes—rice and chips—is performed using keywords ('rice', 'biryani', 'chips', 'fries') identified in the dish names and ingredients. The dataset can be accessed and downloaded directly into a Python environment using the provided code snippet:
```python
!pip install mlend
from mlend import download_yummy
drive.mount('/content/drive')
baseDir = download_yummy(save_to='/path/to/save')
```
This code ensures that users can easily obtain and use the dataset for machine learning or other analytical purposes.

### Machine Learning Pipeline
The pipeline begins with the loading of images, which are then dynamically filtered based on keyword matches to identify dishes containing either rice or chips. Images are resized to 200x200 pixels and normalized. A LinearSVC model is employed to distinguish between the two classes based on extracted features such as color and texture. The pipeline aims for efficiency in processing and accuracy in classification.

### Transformation Stage
Images undergo a series of preprocessing steps including resizing and the extraction of features like the yellow color component and GLCM texture properties. These transformations standardize the input data and enrich it with features pertinent for effective classification.

### Modeling
A LinearSVC model is chosen for its efficacy in binary classification tasks. It is fine-tuned to balance classification accuracy and error minimization, trained using features extracted during the transformation stage.

### Results and Discussion
The LinearSVC model achieved a training accuracy of 84.37% and a testing accuracy of 82.83%. These results demonstrate the model’s ability to effectively classify the images as either containing rice or chips, with reasonable precision.

### Conclusions
The project highlights the potential of LinearSVC for image-based classification tasks. Future enhancements could include exploring more complex models or combining different types of data to improve accuracy. The current pipeline serves as a robust basis for further exploration and refinement.
