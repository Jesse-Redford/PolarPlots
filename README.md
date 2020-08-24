# Polar Plot Project

### Project Description
This project is in Joint effort with Dr.Mullany from the Mechanical Engineering Deparment and the University of North Carolina Charlotte.
The scope of the project is to develop a method for characterizing material surfaces and provide a stastitical measure for estimating surface defects and quality using digital imagining. The repository includes beta versions of the desktop app which can be run on any windows machine with a camera. In addition we offer a library of modules that can be used to develop, modify, and integrate polar plotting into specific applications. If you want to test our a basic implentation download 

### Contributors
- Jesse Redford
- Bridgit Mullany

### Beta Versions of PolarPlot Windows Application

- PolarPlot-v0.exe - basic version of app, capture images from camera stream and generates polar plot
- PolarPlot-v1.exe - includes realtime filter options, livestream comparision, and ability to take screen shots

### Preformance Chart Processing Time 

![PolarPlot](https://github.com/Jesse-Redford/PolarPlots/blob/master/gussian_surface.png)



### On going Research and Development

- Currently the algorith is able to We aim to reduce the processing time required to generate a polar plot 

- This tool could likley be used in fast pass/fail quality inscpections of textured surfaces. 

- To compensate for lighting conditions and other external factors which may effect the surface image, version 1 of the application includes filtering options which can be modifed by the user in realtime. The new version also displays a comparision of the raw and filiterd images, in addition to there polar plots for comparison.

- Although a model can be trained to detect surface defects, the question remains on wether there is enough information stored in the polar plot for a classifer to have the ability to regenerate what the captured surface looks like using only information from the polarplot.




## Polar Plot Examples
Assuming that a surface containing no defects can be described as a gussian surface, the resulting polar plot of this surface should appear as a relativley uniform circle.
However, in the presence of defects the polar plot should take on some new charactersitic.

### Polar Plot of Gussian Surface
![PolarPlot](https://github.com/Jesse-Redford/PolarPlots/blob/master/gussian_surface.png)

### Polar Plot of Gussian Surface with 3 vertial scratches
![PolarPlot](https://github.com/Jesse-Redford/PolarPlots/blob/master/gussian_surface_with_defects.png)

### Polar Plot of Real Gussian Surface with 3 vertial scratches, with and without filtering
![PolarPlot](https://github.com/Jesse-Redford/PolarPlots/blob/master/real_gussian_surface_with_defects.png)



