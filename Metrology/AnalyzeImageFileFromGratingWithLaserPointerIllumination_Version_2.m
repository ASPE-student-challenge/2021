close all;
clearvars;
% Tim Dalrymple
% 4/30/2020
% COVID 19 Days
% Intent: Plot diffraction pattern intensity

set(0, 'DefaultAxesFontSize', 14);

path                    =  'C:\Users\dalry\Google Drive\ASPE\2020 Student Challenge\Metrology';
filename                = 'IPhone7Auto.jpg';
%filename               = 'NikonD7500Nikkor40mm1-2_8.JPG';
filename                = 'NikonSetupDarkroomScreenNotFixedTest.JPG';
filename                = 'NikonSetupDarkroomFixedScreenLaserPointer.JPG';
filename                = 'Iphone7LaserPointerFixedScreen1.jpg';
filename                = 'RedGreenSymSetup275mmZ (4).JPG';
filename                = 'RedGreenSymSetup275mmZ (5).JPG';
filename                = 'RedGreenSymSetup275mmZ (6).JPG';
% filename              = 'RedGreenSymSetup275mmZ (7).JPG';
% filename              = 'RedGreenSymSetup275mmZ (8).JPG';
 filename               = 'RedGreenSymSetup275mmZ (9).JPG';

filename_withPath       = strcat(path, '\', filename);
info                    = imfinfo(filename_withPath);
[A,map]                 = imread(filename_withPath);

% Show the diffraction image on the screen
figure(1);
image(A);
ylabel('Y Camera Axis');
xlabel('X Camera Axis');
xlim([0 size(A,2)]);
title( ['Diffraction Image on Screen for', ' ' , filename]);

% To quantify the diffraction image we convert to gray scale and extract
% the intensity which we normalize
I_grey_2D           = rgb2gray(A);
I_grey_1D           = sum(I_grey_2D);
I_grey_1D_Norm      = I_grey_1D/ max(I_grey_1D);

figure(2);
plot(I_grey_1D_Norm);
ylabel('Normalized Intensity: Integrated along Y Axis');
xlabel('X Axis Pixel Count');
xlim([0 size(A,2)]);
title(['Normalized X Pixel Intensity for', ' ', filename]);

figure(3);
histogram(A);
title(['Histrogram for RGB Image-Check Saturation of Bins for', ' ', filename]);

% Predict the Diffraction Pattern based on Goodman: Introduction for
% Fourier Optics, Joesph W. Goodman 3rd Edition

% I=( A Jq (m2) / (  z))2
 
 % Calculate the Fourier Coefficient of the Grating Physical Geometery.
 % This is readily done by taking the Fourier Tranform of a cross cut of the grating surface
 
 % Assume that the coefficents of an arbitrary grating geometery are as follows
 Amp                  =   1;
 m                  =   50e-9;                                  % Grating Amplitude 
 f_0                =   150e3;                                  % Grating lines per meter frequency zero
 lambda_green       =   532e-9;                                 % Wavelength of green laser pointer from specifications with range of +- 10 nm
 lambda_red         =   (630e-9+680e-9)/2;                      % Average wavelength of red laser pointer from specifions with min/max per label
                                      
 FT_Surf            =   m * [ .8 .7 .6 .5 .4 .3 .05 ];          % Assume that this represents the Fourier Coefficients of the Grating Surface
 
 % Domain
 N_x                =   1000;                                   % Number of points in the x direction (along diffraction pattern)
 z                  =   175e-3;                                 % Distance from grating to screen
 y                  =   0;                                      % Look at the center of the pattern
 x                  =   linspace(-100e-3, 100e-3, N_x);         % This represents the screen extent in the x direction (x refers here to the direction of the spots spreading)
 x_index            =   linspace(1,N_x, N_x);
 w                  =   4e-3;                                   % Grating width in the x direction (square)

 % Setup for this run
 lambda             =   lambda_red;                             % Pick the appropiate wavelenght for the data set         
 N_orders           =   100;                                    % order

 % Preallocate arrays to supress MATLAB warnings
 %q_index            =   zeros(1, N_orders);
 I                  =   zeros(N_orders, N_x);

 for q = 0: N_orders-1                                            % loop through the orders (q) that are desired to represent the signal
    q_index( q )    =   q +1;                                        % need a index for referencing variables later
    I ( q + 1, x_index )=   ( Amp /( lambda * z ) )^2 * besselj ( q, ( m/2 ) )^2 * sinc( (( 2 * w  ) / ( lambda * z) ) * ( x - ( q  * f_0 * lambda * z ) ).^2 * sinc( 2 * w * y / ( lambda * z ) )^2;  
  end
  
 I_sum              = sum(I,1);                                 % Sum all orders to construct the diffraction pattern
 figure(4);
 plot( x*1000,I_sum');
 
 ylabel('Normalized Intensity: Integrated along Y Axis');
 xlabel('X position of diffraction pattern on Screen (mm)');
ylim([0 1.5e7]);
title('Goodman Model for Diffraction Pattern at 150 for a Pure Sinusoidal Grating');




