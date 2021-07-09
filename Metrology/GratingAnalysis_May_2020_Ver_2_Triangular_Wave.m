close all;
clearvars;
% Tim Dalrymple
% 4/30/2020
% Updated 19 June 2020
% COVID 19 Days
% Intent: Plot diffraction pattern intensity
% Model expected intensity for the student challenge grating using Goodman's work

set(0, 'DefaultAxesFontSize', 17);

filename                                = 'IPhone7Auto.jpg';
%filename                               = 'NikonD7500Nikkor40mm1-2_8.JPG';
filename                                = 'NikonSetupDarkroomScreenNotFixedTest.JPG';
filename                                = 'NikonSetupDarkroomFixedScreenLaserPointer.JPG';
filename                                = 'Iphone7LaserPointerFixedScreen1.jpg';
filename                                = 'RedGreenSymSetup275mmZ (4).JPG';
filename                                = 'RedGreenSymSetup275mmZ (5).JPG';
filename                                = 'RedGreenSymSetup275mmZ (6).JPG';
% filename                              = 'RedGreenSymSetup275mmZ (7).JPG';
% filename                              = 'RedGreenSymSetup275mmZ (8).JPG';
 filename                               = 'RedGreenSymSetup275mmZ (9).JPG';

filename_withPath                       = filename;                 % For this to find the file, the folder  were the files are located must be added in the Matlab toolbar.   Home, Set Path
info                                    = imfinfo(filename_withPath);
[A,map]                                 = imread(filename_withPath);

% Show the diffraction image on the screen
figure(1);
image(A);
ylabel('Y Camera Axis');
xlabel('X Camera Axis');
xlim([0 size(A,2)]);
title( ['Diffraction Image on Screen for', ' ' , filename]);

% To quantify the diffraction image we convert to gray scale and extract the intensity which we normalize
I_grey_2D                               = rgb2gray(A);
I_grey_1D                               = sum(I_grey_2D);
I_grey_1D_Norm                          = I_grey_1D/ max(I_grey_1D);

figure(2);
plot(I_grey_1D_Norm);
ylabel('Normalized Intensity: Integrated along Y Axis');
xlabel('X Axis Pixel Count');
xlim([0 size(A,2)]);
title(['Normalized X Pixel Intensity for', ' ', filename]);

% figure(3);
% histogram(A);
% title(['Histrogram for RGB Image-Check Saturation of Bins for', ' ', filename]);
 
% Calculate the Fourier Coefficient of the Grating Physical Geometery by taking the Fourier Tranform of the grating surface
Input_Intensity                         =   1;                                              % Intensity of incident beam
m                                       =   (4.142e-6)/2;                                   % Sawtooth Grating Amplitude -- should be converted to raidans? but this is only a scale see page 83 Goodman--MUST BE NORMALIZED FOR USE IN Bessel Function: This is done at point of implementation in code below
f_grating                               =   75e3;                                           % Grating lines per meter 
lambda_green                            =   532e-9;                                         % Wavelength of green laser pointer from specifications with range of +- 10 nm
lambda_red                              =   ( 630e-9 + 680e-9 ) / 2;                        % Average wavelength of red laser pointer from specifions with min/max per label
lambda                                  =   lambda_red;                                     % Wavelength of Laser illumination         
GrooveDepth_Normal                      =   2 * m / lambda;                                 % This allowes comparision with normalized diffraction plots from Harvey
N_x                                     =   1000;                                           % Number of points in the x direction (along diffraction pattern)--if not equal to 1000, then everything is wonky (single peak in diffraction pattern)
z                                       =   175e-3;                                         % Distance from grating to screen
y                                       =   0;                                              % Look at the center of the pattern
x                                       =   linspace(-100e-3, 100e-3, N_x);                 % This represents the screen extent in the x direction (x refers here to the direction of the spots spreading)
x_index                                 =   linspace(1,N_x, N_x);                           % Need the index for x as an array for manipulation below
w_x                                     =   4e-3;                                           % Grating width in the x direction // Code only verfified for square grids
w_y                                     =   4e-3;                                           % Grating width in the y direction // Code only verfified for square grids
 
% Create an Array of magnitudes (m) to represent the Fourier Transform of a grating that is not a pure sinusoid
Grating_width                           =   w_x;                                                    % Physical length of signal (grating aperture) in meters
Sample_Freq                             =   10e6;                                                   % Sampling Frequency of this wave: samples/meter: user's choice with computational overhead
Sample_X                                =   0 : 1 / Sample_Freq : Grating_width - 1 / Sample_Freq;  % X coordinate of sample point locations in meters
Profile                                 =   m * sawtooth( 2 * pi * f_grating * Sample_X, 1/2);      % This generates a exact 135 degree angle sawtooth with no radius on the tool
 
figure(4);
plot   (Sample_X * 1e3, Profile * 1e6);
ylabel ('Nominal Grating Surface Height (\mum)');
xlabel ('X Dimension of Grating (mm)');
title  ('Nominal Surface Model -- Simple Sawtooth: refine to 135 angle');
 
% Fourier Transform the Profile to get the coefficients of a Sine Wave Implementation for the Goodman Equations
delta_F                                 = Sample_Freq / length( Profile );                  % Frequency resolution: lines per meter
Nyquist_Freq_num_pts                    = length( Profile )/ 2;
Surface_FT                              = abs( fft( Profile ) );                            % Amplitude correct without additional normalization
index                                   = [1 : Nyquist_Freq_num_pts];
Freq                                    = (index - 1) * delta_F;     
Surface_FT_SS                           = Surface_FT( 1 : Nyquist_Freq_num_pts) / Nyquist_Freq_num_pts;     % Single Sided Fourier Transform with Correct Normalization of Amplitude
 
figure(5); 
stem((Freq)/1000, Surface_FT_SS * 1e6);
xlabel ('Spatial Frequency (1/mm)');
ylabel ('Amplitude (\mum)');
title  ('Nominal Frequency Content of Diffraction Grating Geometry');
ylim([0 2]);
xlim([0 1000]);
 
Amplitude_threshold                     =   10e-9;                                          % Neglect Fourier coifficients/terms smaller than this value
Harmonics_Sig                           =   Surface_FT_SS > Amplitude_threshold;            % Returns 1 or zero: boolean element by element test of amplitude significance 
Surface_FT_SS_Clean                     =   Surface_FT_SS .* Harmonics_Sig;                 % If the amplitude is less than threshold then keep the value, otherwise set it to zero
Freq_Clean                              =   Freq .* Harmonics_Sig;                          % If the amplitude is less than threshold then keep the value, otherwise set it to zero
Surface_FT_SS_No_Zeros                  =   nonzeros(Surface_FT_SS_Clean);                  % Extract the Amplitude of the valued Fourier Components
Freq_Clean_No_Zeros                     =   nonzeros(Freq_Clean);                           % Extract the Frequencies of the valued Fourier Components
Number_of_Sig_Har                       =   length( Surface_FT_SS_No_Zeros );
I_Single_Freq_all_orders                =   zeros ( Number_of_Sig_Har, N_x );
 
% Predict the Diffraction Pattern based on Goodman: Introduction for  Fourier Optics, Joesph W. Goodman 3rd Edition
 
% for h = 1: Number_of_Sig_Har                                                                % Sum the contribution of all the single frequency sine diffraction gratings that comprise the trianglular wave form
%     N_orders                            =   floor ( 2 / ( lambda * h * f_grating ) );       % Limit the calcualtions to the realizable orders. 
%     T1                                  =   zeros ( N_orders, N_x );                        % Allocate memory for the terms of the intensity equation due to complexity
%     T2                                  =   zeros ( N_orders, N_x );
%     T3_Pos_Ord                          = 	zeros ( N_orders, N_x );
%     T3_Neg_Ord                          =   zeros ( N_orders, N_x );
%     T4                                  =   zeros ( N_orders, N_x );
%     I                                   =   zeros ( N_orders, N_x );                        % This matrix sums the diffraction pattern over the specified number of orders for a single frequency sine wave grating
%     
%     for q                               = 0: N_orders-1                                     % Dual Loop: x domain and all orders (q) simultaneously: This is calculated above as N_orders (higher orders are not physically realizable
%         T1( q + 1, x_index )            = ( Input_Intensity / ( lambda * z ) )^2;           % Calculate the Intensity Equation on a term-by-term basis
%         T2( q + 1, x_index )            = besselj ( q, ( ( 4 * pi / lambda) * Surface_FT_SS_No_Zeros( h ) / 2 ) )^2;
%         T3_Pos_Ord( q + 1, x_index )    = sinc( ( ( 2 * w_x  ) / ( lambda * z ) ) * ( x - ( +q  * Freq_Clean_No_Zeros ( h ) * lambda * z ) ) ).^2;
%         T3_Neg_Ord( q + 1, x_index )    = sinc( ( ( 2 * w_x  ) / ( lambda * z ) ) * ( x - ( -q  * Freq_Clean_No_Zeros ( h ) * lambda * z ) ) ).^2;               % Shortcut to include the negative orders without doubling the loops
%         T4( q + 1, x_index )            = ( sinc( 2 * w_x * y  / ( lambda * z ) )^2 );                                                                           % if you divide by N_orders, it make it look like the code is working by reducing the amplitude of higher frequency components since they have fewer orders
%         I ( q + 1, : )                  = T1( q + 1, x_index )  .* T2( q + 1, : ) .* ( T3_Pos_Ord( q + 1, : ) + T3_Neg_Ord( q + 1, : ) ) .* T4( q + 1, :); % Combine the terms    
%     end
%      I_Single_Freq_all_orders( h, : )    = sum( I , 1 );                                     % Sum all orders to construct the diffraction pattern for a single frequency sine wave grating Consruct matrix to hold the summed order response for each sine wave frequency 
% end
% 
% I_sum_orders_and_har                    = sum( I_Single_Freq_all_orders, 1 );                 % Sum the response for each of the single frequency sine wave diffraction patterns to get the response to the machined surface (triagular wave)
%  
% figure(6);
% surf( x * 1000, Freq_Clean_No_Zeros / 1000, I_Single_Freq_all_orders );
% title('Intensity Plot v Location of frequency of Fourier Compoents of Triangular Grating Geometry:  Zero Order Supressed');
% ylabel('Sine Wave Frequency (waves/mm)');
% xlabel('X Location (mm)');

% figure(7);
% plot( x * 1000, I_Single_Freq_all_orders );
% title('Nominal Intensity Plot for single frequency:  Zero Order Supressed');
% ylabel('Intensity (no units)');
% xlabel('X Location (mm)');
% ylim([0 5e12]);
%  
% figure(8);
% plot( x * 1000, I_sum_orders_and_har );
% title('Nominal Intensity Plot:  Zero Order Supressed');
% ylabel('Intensity (no units)');
% xlabel('X Location (mm)');
% ylim([0 5e12]);

% Vary the groove depth h(phase delay) Intially we do this incrementally Later, we can do it as a Monte Carlo Simulation
N_amp                                       =   750;                                            % Number of data points along the grating, used to modulate the depth to check Bessel function or effect of errors
Grating_height_ratio                        =   linspace(0.5, 1.1, N_amp);                      % This represents the screen extent in the x direction (x refers here to the direction of the spots spreading)
I_All_v_groove_depth                        =   zeros(N_amp, N_x);
[c, index]                                  = min(abs(Grating_height_ratio-1));                 % Find the index of the nominal Groove depth 

for i_y = 1: N_amp                                                                              % Loop through a variation in Groove Depth to examine the sensitivity to error
    for h = 1: Number_of_Sig_Har                                                                % Sum the contribution of all the single frequency sine diffraction gratings that comprise the trianglular wave form
        N_orders                            =   floor ( 2 / ( lambda * h * f_grating ) );       % Limit the calcualtions to the realizable orders. 
        T1                                  =   zeros ( N_orders, N_x );                        % Allocate memory for the terms of the intensity equation due to complexity
        T2                                  =   zeros ( N_orders, N_x );
        T3_Pos_Ord                          = 	zeros ( N_orders, N_x );
        T3_Neg_Ord                          =   zeros ( N_orders, N_x );
        T4                                  =   zeros ( N_orders, N_x );
        I                                   =   zeros ( N_orders, N_x );                        % This matrix sums the diffraction pattern over the specified number of orders for a single frequency sine wave grating

        for q                               = 0: N_orders-1                                     % Dual Loop: x domain and all orders (q) simultaneously: This is calculated above as N_orders (higher orders are not physically realizable
            T1( q + 1, x_index )            = ( Input_Intensity / ( lambda * z ) )^2;           % Calculate the Intensity Equation on a term-by-term basis
            T2( q + 1, x_index )            = besselj ( q, ( ( 4 * pi / lambda) * Grating_height_ratio (i_y) * Surface_FT_SS_No_Zeros( h ) / 2 ) )^2;
            T3_Pos_Ord( q + 1, x_index )    = sinc( ( ( 2 * w_x  ) / ( lambda * z ) ) * ( x - ( +q  * Freq_Clean_No_Zeros ( h ) * lambda * z ) ) ).^2;
            T3_Neg_Ord( q + 1, x_index )    = sinc( ( ( 2 * w_x  ) / ( lambda * z ) ) * ( x - ( -q  * Freq_Clean_No_Zeros ( h ) * lambda * z ) ) ).^2;               % Shortcut to include the negative orders without doubling the loops
            T4( q + 1, x_index )            = ( sinc( 2 * w_x * y  / ( lambda * z ) )^2 );                                                                           % if you divide by N_orders, it make it look like the code is working by reducing the amplitude of higher frequency components since they have fewer orders
            I ( q + 1, : )                  = T1( q + 1, x_index )  .* T2( q + 1, : ) .* ( T3_Pos_Ord( q + 1, : ) + T3_Neg_Ord( q + 1, : ) ) .* T4( q + 1, :); % Combine the terms    
        end                                                                                     % Diffraction Orders Loop
         
        I_Single_Freq_all_orders( h, : )   = sum( I , 1 );                                      % Sum all orders to construct the diffraction pattern for a single frequency sine wave grating Consruct matrix to hold the summed order response for each sine wave frequency 
        
        if (i_y == index && h == 1)
            I_Single_Freq_all_orders_Nom_grv_dpth   = I_Single_Freq_all_orders( h, : );
        end 
    end                                                                                         % Sine Wave Harmonics Loop
    
    I_All_v_groove_depth ( i_y, : )         = sum( I_Single_Freq_all_orders, 1 );
    
    if (i_y == index)
        I_sum_orders_and_har_Nom_grv_dpth   = I_All_v_groove_depth ( i_y, : );
    end
end

figure(6);
surf( x * 1000, Freq_Clean_No_Zeros / 1000, I_Single_Freq_all_orders );
title('Intensity Plot v Location of frequency of Fourier Compoents of Triangular Grating Geometry:  Zero Order Supressed');
ylabel('Sine Wave Frequency (waves/mm)');
xlabel('X Location (mm)');

figure(9);
z_plot_max = 1e13;
Color_Gain = max(I_All_v_groove_depth, [], 'all') /z_plot_max;
surf( x * 1000,  Grating_height_ratio * ( 2 * m ) * 1e6, I_All_v_groove_depth,  I_All_v_groove_depth / Color_Gain, 'edgecolor','none' );
title('Intensity of Saw Tooth Diffraction Pattern for Different Grating Groove Depths: Zero Order Supressed');
ylabel('Peak to Peak Grating Height (\mum)');
xlabel('X Location (mm)');
zlabel('Diffration Spot Intensity (no units)');

figure(10);
plot( x * 1000, I_All_v_groove_depth (index,:));
title('Intensity of Saw Tooth Diffraction Pattern for Nominal Grating: Zero Order Supressed');
ylabel('Intensity (no units)');
xlabel('X Location (mm)');

% This section will show--for a single spatial frequency sine wave phase grating--how the amplitude is modulated by changing the groove depth.  This shows the sensitivity of
% the diffraction pattern to this manufacturing parameter
 
% Reduce the analyisis to a single frequency sine wave phase grating                        % Neglect Fourier coifficients/terms smaller than this value
Harmonics_Sig                       =   Surface_FT_SS > Amplitude_threshold;                % Returns 1 or zero: boolean element by element test of amplitude significance 
Fundamental_Freq                    =   find(Harmonics_Sig,1,'first');                      % Index for Fundamental Frequency Component
Surface_FT_SS_No_Zeros              =   Surface_FT_SS ( Fundamental_Freq );                 % Get the fundamental frequency ampliude
Freq_Clean_No_Zeros                 =   Freq( Fundamental_Freq );                           % Get the assoicated frequency
N_orders                            =   floor ( 2 / ( lambda * h * f_grating ) );           % Limit the calcualtions to the realizable orders. 

T1                                  =   zeros ( N_orders, N_x );                            % Allocate memory for the terms of the intensity equation due to complexity
T2                                  =   zeros ( N_orders, N_x );
T3_Pos_Ord                          = 	zeros ( N_orders, N_x );
T3_Neg_Ord                          =   zeros ( N_orders, N_x );
T4                                  =   zeros ( N_orders, N_x );
I                                   =   zeros ( N_orders, N_x );                            % This matrix sums the diffraction pattern over the specified number of orders for a single frequency sine wave grating

% Vary the groove depth h(phase delay) Intially we do this incrementally Later, we can do it as a Monte Carlo Simulation
N_orders                            =   floor ( 2 / ( lambda * f_grating ) );               % Single freqency, so the number of orders is fixed 

for i_y = 1: N_amp                                                                          % Loop through a variation in Groove Depth to examine the sensitivity to error
    for q                               = 0: N_orders-1                                     % loop through the orders (q) that exist: This is calculated above as N_orders (higher orders are not physically realizable
        T1( q + 1, x_index )            = ( Input_Intensity / ( lambda * z ) )^2;           % Calculate the Intensity Equation on a term-by-term basis
        T2( q + 1, x_index )            = besselj ( q, ( ( 4 * pi / lambda) * Grating_height_ratio (i_y) * (2 * Surface_FT_SS_No_Zeros) / 2 ) ).^2;
        T3_Pos_Ord( q + 1, x_index )    = sinc( ( ( 2 * w_x  ) / ( lambda * z ) ) * ( x - ( +q  * Freq_Clean_No_Zeros * lambda * z ) ) ).^2;
        T3_Neg_Ord( q + 1, x_index )    = sinc( ( ( 2 * w_x  ) / ( lambda * z ) ) * ( x - ( -q  * Freq_Clean_No_Zeros * lambda * z ) ) ).^2;        % Shortcut to include the negative orders without doubling the loops
        T4( q + 1, x_index )            = sinc( 2 * w_x * y  / ( lambda * z ) )^2 ;                                                                 % if you divide by N_orders, it make it look like the code is working by reducing the amplitude of higher frequency components since they have fewer orders
        I ( q + 1, : )                  = T1( q + 1, x_index ) .* T2( q + 1, : ) .* ( T3_Pos_Ord( q + 1, : ) + T3_Neg_Ord( q + 1, : ) ) .* T4( q + 1, :); % Combine the terms 
    end

    I_Single_Freq_all_orders   ( i_y, : )  = sum( I , 1 );                                  % Sum all orders to construct the diffraction pattern for a single frequency sine wave grating Consruct matrix to hold the summed order response for each sine wave frequency
    T1_all_orders              ( i_y, : )  = sum( T1, 1 );                                  % T1 is not exactly constant across all harmonics as it should be due to varying numbers of orders for different harmnics.  This results in small numerical errors
    T2_all_orders              ( i_y, : )  = sum( T2, 1 );
    T3_Pos_all_orders          ( i_y, : )  = sum( T3_Pos_Ord, 1 );
    T3_Neg_all_orders          ( i_y, : )  = sum( T3_Neg_Ord, 1 );     
end
 
% BesselFunctionDomain = ( 4 * pi / lambda) * Grating_height_ratio * Surface_FT_SS_No_Zeros / 2;  % just checking
figure(11);
Color_Gain = max(I_Single_Freq_all_orders, [], 'all') /z_plot_max;
surf( x * 1000, Grating_height_ratio * ( 2 * Surface_FT_SS_No_Zeros ) * 1e6, I_Single_Freq_all_orders, I_Single_Freq_all_orders / Color_Gain,  'edgecolor','none' );
title('Intensity of Single Sine Wave Diffraction Pattern for Different Grating Groove Depths: Zero Order Supressed');
ylabel('Peak to Peak Grating Height (\mum)');
xlabel('X Location (mm)');
zlabel('Diffration Spot Intensity (no units)');
 




