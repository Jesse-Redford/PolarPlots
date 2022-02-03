import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.color'] = 'white'
mpl.rcParams['image.cmap'] = 'jet'
mpl.rcParams['xtick.color'] ='white'
mpl.rcParams['ytick.color'] ='white'
mpl.rcParams['font.family'] = 'serif'
plt.rcParams['figure.facecolor'] = 'white' #'black'


def level_surface(Y):
    m, m = Y.shape
    X1, X2 = np.mgrid[:m, :m]
    # Regression
    X = np.hstack((np.reshape(X1, (m * m, 1)), np.reshape(X2, (m * m, 1))))
    X = np.hstack((np.ones((m * m, 1)), X))
    YY = np.reshape(Y, (m * m, 1))
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
    plane = np.reshape(np.dot(X, theta), (m, m))
    # return surface with bestfit plane removed
    return Y - plane


class polarplot:
    def __init__(self):
        self.parameters = ["rmin","rmax","rmean","rstd", "rsum","rmin_theta","rmax_theta","num_lobes"]

    def compute(self,z):
        surface = z
        primary_surface = surface  # gaussian_filter(surface, sigma=.2)
        surface = primary_surface - np.mean(surface)
        rq_columns = np.zeros((180, (int(round(len(surface) / np.sqrt(2) / 2))) * 2))
        # Rotate surface through 0 to 180 degrees and record std
        for theta in np.arange(0, 180, 1):
            # Rotate the image
            rotated_surface = cv2.warpAffine(surface,
                                             cv2.getRotationMatrix2D(
                                                 tuple(np.array([surface.shape[0], surface.shape[1]]) / 2),
                                                 -theta, 1.0),
                                             (surface.shape[1], surface.shape[0]),
                                             flags=cv2.INTER_NEAREST)

            # Calcs stedev of each column in a submatrix of the rotated matrix and store the returned array into the ith row index of the rq_columns matrix
            rq_columns[theta, :] = np.std(rotated_surface[
                                          int(np.floor(len(rotated_surface) / 2)) - int(
                                              round(len(surface) / np.sqrt(2) / 2)):int(
                                              np.floor(len(rotated_surface) / 2)) + int(
                                              round(len(surface) / np.sqrt(2) / 2)),
                                          int(np.floor(len(rotated_surface) / 2)) - int(
                                              round(len(surface) / np.sqrt(2) / 2)):int(
                                              np.floor(len(rotated_surface) / 2)) + int(
                                              round(len(surface) / np.sqrt(2) / 2))],
                                          axis=0)

        self.radii = rq_columns
        self.radii = np.std(rq_columns.T, axis=0)
        return self.radii

    def rmin(self):
        return self.radii.min()

    def rmax(self):
        return self.radii.max()

    def rmean(self):
        return self.radii.mean()

    def rstd(self):
        return self.radii.std()

    def rsum(self):
        return self.radii.sum()

    def rmin_theta(self):
        return np.argmin(self.radii)

    def rmax_theta(self):
        return np.argmax(self.radii)

    def num_lobes(self):
        return int(
            len(np.argwhere(np.diff(np.sign(np.full_like(self.radii, self.radii.mean()) - self.radii))).flatten()) / 2)

    def invalid_op(self):
        raise Exception("Invalid operation, parmaeter is not in parameter lib or no parameter method was defined")

    def parameter_lib(self, chosen_parameter):
        lib = {"invalid_op": self.invalid_op,
               "rmin": self.rmin,
               "rmax": self.rmax,
               "rmean": self.rmean,
               "rstd": self.rstd,
               "rsum": self.rsum,
               "rmin_theta": self.rmin_theta,
               "rmax_theta": self.rmax_theta,
               "num_lobes": self.num_lobes
               }
        chosen_parameter_function = lib.get(chosen_parameter, self.invalid_op)
        return chosen_parameter_function()

    def features(self, z):
        self.compute(z)
        parameters_name_value = []
        for parameter in self.parameters:
            parameters_name_value.append((parameter, self.parameter_lib(chosen_parameter=parameter)))
        return parameters_name_value

    def plot(self,z):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(20, 10))
        ax.set_theta_zero_location("N")

        self.thetas = np.linspace(0, ((2 * 180) - 1) * np.pi / 180, 360) #np.arange(0, 360, 1)

        self.compute(z)

        self.radii = np.concatenate((self.radii , self.radii ), axis=0)


        plt.polar(self.thetas, self.radii, color='black')

        T = np.linspace(0, ((2 * 180) - 1) * np.pi / 180, 360)

        # Compute number of lobes that appears in reference to r avg +- std
        Upper_Lobes = int(len(np.argwhere(np.diff(np.sign(self.radii - self.radii.mean()+self.radii.std()))).flatten()) / 2)
        Lower_Lobes = int(len(np.argwhere(np.diff(np.sign(self.radii.mean()-self.radii.std() - self.radii))).flatten()) / 2)

        # Compute total number of lobes based on r avg reference
        Lobes = int(len(np.argwhere(np.diff(np.sign(np.full_like(self.radii, self.radii.mean()) - self.radii))).flatten()) / 2)

        # Estimate number of scratches on surface
        est_num_scratches = int(Lobes / 2)



        ZERO_BOUND = np.empty(len(T));
        ZERO_BOUND.fill(0)
        MIN_RADIUS = np.empty(len(T));
        MIN_RADIUS.fill(np.min(self.radii))
        MAX_RADIUS = np.empty(len(T));
        MAX_RADIUS.fill(np.max(self.radii))

        MEAN_RADIUS = np.empty(len(T));
        MEAN_RADIUS.fill(np.mean(self.radii))
        NEG_STD_RADIUS = np.empty(len(T));
        NEG_STD_RADIUS.fill(np.mean(self.radii) - np.std(self.radii))
        POS_STD_RADIUS = np.empty(len(T));
        POS_STD_RADIUS.fill(np.mean(self.radii) + np.std(self.radii))

        PLOT_LIMIT = np.empty(len(T));
        PLOT_LIMIT.fill(np.max(self.radii) + np.mean(self.radii) * 2)

        LOBAL_LOWER_LIMIT = np.empty(len(T));
        LOBAL_LOWER_LIMIT.fill(np.max(self.radii) + np.std(self.radii))
        LOBAL_UPPER_LIMIT = np.empty(len(T));
        LOBAL_UPPER_LIMIT.fill(np.max(self.radii) + np.std(self.radii) * 2)


        plt.polar(T, MEAN_RADIUS, color='olive', linestyle='--', alpha=.75, linewidth=1,
                  label='r_mean:' + ' ' + str(self.radii.mean()))

        plt.polar(T, MIN_RADIUS, color='black', linestyle='--', alpha=.75, linewidth=1,
                  label='Min Radius:' + ' ' + str(self.radii.min()))
        plt.polar(T, MAX_RADIUS, color='black', linestyle='-.', alpha=.75, linewidth=1,
                  label='Max Radius' + ' ' + str(self.radii.max))

        plt.fill_between(T, NEG_STD_RADIUS, POS_STD_RADIUS, facecolor='darkgreen', edgecolor='black', interpolate=True,
                         alpha=.25, label='mean+-std' + ' ' + str(self.radii.mean()) + '+-' + ' ' + str(self.radii.std()))


        plt.fill_between(T, ZERO_BOUND, NEG_STD_RADIUS, facecolor='black', alpha=.125)
        plt.fill_between(T, POS_STD_RADIUS, PLOT_LIMIT, facecolor='black', alpha=.125)

        plt.fill_between(T, NEG_STD_RADIUS, self.radii, where=self.radii < NEG_STD_RADIUS, facecolor='navy', edgecolor='black',
                         alpha=.75, label='# Lower Outliers:' + ' ' + str(Lower_Lobes))
        plt.fill_between(T, POS_STD_RADIUS, self.radii, where=self.radii > POS_STD_RADIUS, facecolor='yellow', edgecolor='black',
                         alpha=.75, label='# Upper Outliers:' + ' ' + str(Upper_Lobes))
        plt.fill_between(T, POS_STD_RADIUS, self.radii, where=self.radii > POS_STD_RADIUS, facecolor='black', edgecolor='black',
                         alpha=.25)

        plt.fill_between(T, LOBAL_LOWER_LIMIT, LOBAL_UPPER_LIMIT, where=self.radii > MEAN_RADIUS , facecolor='red',
                         edgecolor='black', alpha=.9,
                         label='# Lobes/Scratches:' + ' ' + str(Lobes) + '/' + str(est_num_scratches))

        plt.fill_between(T, LOBAL_LOWER_LIMIT, LOBAL_UPPER_LIMIT, where=self.radii < MEAN_RADIUS, facecolor='black',
                         alpha=.5)

        leg = plt.legend(bbox_to_anchor=(1.125, -.065), ncol=4, fancybox=True, shadow=True, prop={'size': 6},
                         framealpha=.5)

        for text in leg.get_texts():
            plt.setp(text, color='k')
        plt.ylim([0, np.max(self.radii) + np.std(self.radii) * 2])

        # plt.yticks(np.arange(r_min, r_max, 1.0))

        #plt.yticks(np.arange(0, r_max, 1.0))

        # only show callout for specfic self.radii values
        # plt.yticks([np.min(self.radii),np.mean(self.radii)-np.std(self.radii),np.mean(self.radii),np.mean(self.radii)+np.std(self.radii),np.max(self.radii)])

        plt.grid(linewidth=1)
        plt.show()

        return fig


import PIL
from PIL import Image, ImageOps

st.title('Polar Plots')
pp = polarplot()
uploaded_file = img= st.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg', 'bmp', 'tif'])
image = Image.open(uploaded_file)

image = ImageOps.grayscale(image)
image = np.asarray(image).astype('uint8')
print(image.shape)
m,n = image.shape
image = np.resize(image,(np.min([m,n]),np.min([m,n])))


col1,col2 = st.columns(2)
col1.header('Surface Image')
col2.header('PolarPlot')
col1.image(image)



#img = cv2.imread(uploaded_file, cv2.IMREAD_GRAYSCALE)
col2.pyplot(pp.plot(image))
col2.dataframe(pd.DataFrame(pp.features(image)))




#if __name__ == '__main__':
#    print('PolarPlot')
#    pp = polarplot()
#    print('Number of Parameters', len(pp.parameters))
#    print(pp.plot(np.random.randint(-255, 255, (100, 100))))