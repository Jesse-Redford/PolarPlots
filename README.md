# PolarPlots

### Project Description
This project is in joint effort with Dr.Mullany from the Mechanical Engineering Deparment and the University of North Carolina Charlotte.
The scope of the project is to develop a method for characterizing material surfaces and provide a stastitical measure for estimating surface defects using digital imagining. The repository includes beta versions of the desktop app which can be run on any windows machine with a camera. In addition we offer a library of modules that can be used to develop, modify, and integrate polar plotting into specific applications. If you have an application that may benfit from this form of anaylsis and would like to dicuss development options, feel free to contact one of the project contribtors for consulation.

##### Contributors
- Jesse Redford | PhD student Dept. Mechanical Engineering UNCC | email: jesse.k.redford@gmail.com  <!-- https://jessekredford.wixsite.com/jesseredford -->
- Dr.Bridgit Mullany | Professor Dept. Mechanical Engineering UNCC  | email: bamullan@uncc.edu  <!--https://coefs.uncc.edu/bamullan/ -->


##### PolarPlot Apps for Windows (requires windows OS and Camera)
- PolarPlot-v0.exe (avaliable) - basic version of app, capture images from camera stream and generates polar plot
- PolarPlot-v1.exe (avaliable) - realtime filter options, livestream comparision, screen shot 
- PolarPlot-v1.1.exe (in development) - integrates FFT filter for removing low freqency content from image surface 
- PolarPlot-v1.2.exe (in development) - allows for time series processing for rolling opperations

#### Interperation and Examples of PolarPlots for Surface Anaylsis 
- Assuming that a surface containing no defects can be described as a gussian surface, the resulting polar plot of this surface should appear as a relativley uniform circle.
However, in the presence of defects the polar plot should take on some new charactersitic. 

- Polar Plots | Gussian Surface | Gussian Surface with 3 vertial scratches

<img src="https://github.com/Jesse-Redford/PolarPlots/blob/master/gussian_surface.png" width="450" height="250"> <img src="https://github.com/Jesse-Redford/PolarPlots/blob/master/gussian_surface_with_defects.png" width="450" height="250"> 

<!--  | Real Gussian Surface with 3 vertial scratches, with and without filtering <img src="https://github.com/Jesse-Redford/PolarPlots/blob/master/real_gussian_surface_with_defects.png" width="250" height="250"> 
##### Requirments
- Desktop application - windows OS and Camera 
- API - pip install requirments.txt
![PolarPlot](https://github.com/Jesse-Redford/PolarPlots/blob/master/Processing_Analysis_time_vs_image_size.png)
-->

#### Preformance Reference 
- Depending on your application and the level of resolution, the chart below outlines the approximate processing time to generate a polar plot for various image sizes.

<img src="https://github.com/Jesse-Redford/PolarPlots/blob/master/Processing_Analysis_time_vs_image_size.png" width = "1200" height="350">

##### Research and Development
- This tool could likley be used in fast pass/fail quality inscpections of textured surfaces. 
- To compensate for lighting conditions and other external factors which may effect the surface image, we are currently integrating various filtering options which can be modifed by the user in realtime. The new versions of the application also displays a comparision of the raw and filiterd images, in addition to there polar plots for comparison.
- Although a model can be trained to detect surface defects, the question remains on wether there is enough information stored in the polar plot for a classifer to have the ability to regenerate what the captured surface looks like using only information from the polarplot.

##### Refrences 
- Polar plots for surface characterization and defect detection https://coefs.uncc.edu/bamullan/files/2020/05/Farzad-defense_public.pdf
- International Standards for Surface Texture https://guide.digitalsurf.com/en/guide-filtration-techniques.html





