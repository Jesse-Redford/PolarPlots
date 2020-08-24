# Polar Plot Project

### Project Description
This project is in Joint effort with Dr.Mullany from the Mechanical Engineering Deparment and the University of North Carolina Charlotte.
The scope of the project is to develop a method for characterizing material surfaces and provide a stastitical measure for estimating surface defects and quality using digital imagining. The repository includes beta versions of the desktop app which can be run on any windows machine with a camera. In addition we offer a library of modules that can be used to develop, modify, and integrate polar plotting into specific applications. 

### Contributors
- Jesse Redford
- Bridgit Mullany

### Download Beta Versions of PolarPlot Windows Application 
- PolarPlot-v0.exe - basic version of app, capture images from camera stream and generates polar plot
- PolarPlot-v1.exe - includes realtime filter options, livestream comparision, and ability to take screen shots

### Preformance Chart Processing Time 
Depending on your application and the level of resolution, the chart below outlines the approximate processing time to generate a polar plot for various image sizes.
![PolarPlot](https://github.com/Jesse-Redford/PolarPlots/blob/master/Processing_Analysis_time_vs_image_size.png)


### Research and Development

- This tool could likley be used in fast pass/fail quality inscpections of textured surfaces. 

- To compensate for lighting conditions and other external factors which may effect the surface image, we are currently integrating various filtering options which can be modifed by the user in realtime. The new versions of the application also displays a comparision of the raw and filiterd images, in addition to there polar plots for comparison.

- Although a model can be trained to detect surface defects, the question remains on wether there is enough information stored in the polar plot for a classifer to have the ability to regenerate what the captured surface looks like using only information from the polarplot.



## Polar Plot Examples
Assuming that a surface containing no defects can be described as a gussian surface, the resulting polar plot of this surface should appear as a relativley uniform circle.
However, in the presence of defects the polar plot should take on some new charactersitic. 

- Polar Plot of Gussian Surface
- Polar Plot of Gussian Surface with 3 vertial scratches
- Polar Plot of Real Gussian Surface with 3 vertial scratches, with and without filtering

<img src="https://github.com/Jesse-Redford/PolarPlots/blob/master/gussian_surface.png" width="250" height="250"> <img src="https://github.com/Jesse-Redford/PolarPlots/blob/master/gussian_surface_with_defects.png" width="250" height="250"> <img src="https://github.com/Jesse-Redford/PolarPlots/blob/master/real_gussian_surface_with_defects.png" width="250" height="250">

