# Content Based Image Retrieval

## About
Master thesis project which implements and tests various different methods on image retireval.
Developed web application that retrieves the most similar images based on neural network embeddings of each image.

## Web app view
<img src="https://github.com/adagymnast/ContentBasedImageRetrieval/blob/master/Images/Image%20retrieval%200.png" width="563" height="303">
<img src="https://github.com/adagymnast/ContentBasedImageRetrieval/blob/master/Images/Image%20retrieval%201.png" width="563" height="303">

## Full text
Full text available here:
https://dspace.cuni.cz/bitstream/handle/20.500.11956/173675/120418791.pdf?sequence=1&isAllowed=y

## Abstract
The Wienbibliothek im Rathaus, Vienna City Library, collected over
300 thousand posters scanned in high quality from the last 100 years. Browsing
and searching in such a large dataset is beyond human power. Therefore,
a project was set up in cooperation with the Technical University of Vienna to
test the possibilities of automatic data annotation on a selected sample. One of
the requirements was Content-based Image Retrieval - retrieving images based
on their visual content. This thesis reviews these techniques that emerged over
the last decades. We focus on simple techniques based on colour, texture, and
shape, as well as more advanced algorithms using convolutional neural networks.
We implement these methods and compare their retrieval effectiveness on particular
image datasets. Finally, we describe the functionality of a developed web
application.

## Keywords
Content-Based Image Retrieval, Image Features, Convolutional Neural
Networks, Transfer Learning.

## Functionality

Content-based image retrieval (CBIR) represents a technique to extract image
features based on visual content. In other words, each image is indexed based on
its visual properties, like colour, texture and shape. The main goal of CBIR is to
find the most similar images to an image defined by a user from a given database.
Therefore, the images need to be characterized efficiently to keep similar images
close in terms of distance.

<img src="https://github.com/adagymnast/ContentBasedImageRetrieval/blob/master/Images/CBIR%20system.PNG" width="500" height="350">

## Results

### Results of tSNE of embeddings

Wang dataset

<img src="https://github.com/adagymnast/ContentBasedImageRetrieval/blob/master/Images/tsne_wang.PNG" width="500" height="350">

Patterns dataset

Results after finetuning.
<img src="https://github.com/adagymnast/ContentBasedImageRetrieval/blob/master/Images/tsne_patterns.PNG" width="500" height="350">

### Finetuning loss curve on patterns dataset
<img src="https://github.com/adagymnast/ContentBasedImageRetrieval/blob/master/Images/loss_patterns.PNG" width="500" height="350">
