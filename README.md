# Content Based Image Retrieval

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

![Content-Based Image Retrival Pipeline](https://github.com/adagymnast/ContentBasedImageRetrieval/blob/master/Images/CBIR%20system.PNG | width=100)
