%   Load the example human brain data for your BMDE 660 assignment
input   =   matfile('data.mat');
truth   =   input.truth;
calib   =   input.calib;

%   ============================================================================
%   The R=3 problem - this is already coded. Show results for Question 6a
%   ============================================================================

R       =   [1,3];
kernel  =   [3,4];

mask    =   false(32,96,96);
mask(:,:,1:3:end)   =   true;

data    =   truth.*mask;

recon   =   grappa(data, calib, R, kernel);

show_quad(data, recon, 'R=3');


%   ============================================================================
%   For BMDE 660:   
%
%   Create more realistic, noisy R=3 data and reconstruct images. 
%   Specifically, use complex data with SNR values of 123, 120, 112, 106, 103 and 
%   100 dB respectively. Examine the image domain SNR.
%   ============================================================================

% Define SNR values (in dB)
SNRdB = [123, 120, 112, 106, 103, 100];
%SNRdB = [123];

% Preallocate arrays for storing SNR values
imageSNR = zeros(1, length(SNRdB));

% Loop over each SNR level
for i = 1:length(SNRdB)
    % Add noise to the complex data
    I_noise = awgn(truth, SNRdB(i));
    I_noise = I_noise .* mask;
    
    % Apply GRAPPA reconstruction
    recon = grappa(I_noise, calib, R, kernel);
    
    % Display images
    figure();
    set(gcf, 'Renderer', 'painters');
    
    % First subplot - Noisy Image
    subplot(1, 2, 1);
    imshow(squeeze(sum(abs(ifftdim(I_noise,2:3)).^2,1)).^0.5', []);
    title("Noisy Image (SNR: " + string(SNRdB(i)) + " dB)", ...
          'Units', 'normalized', 'Position', [0.5, 1.05, 0]); % Fixed position
    set(gca, 'YDir', 'normal');
    
    % Second subplot - GRAPPA Reconstruction
    subplot(1, 2, 2);
    recon_image = squeeze(sum(abs(ifftdim(recon,2:3)).^2,1).^0.5)';
    imshow(recon_image, []);
    title("GRAPPA Reconstruction", ...
          'Units', 'normalized', 'Position', [0.5, 1.05, 0]); % Fixed position
    set(gca, 'YDir', 'normal');
    
    % Define ROIs
    N = floor(12/2);  % Half-width of the ROI
    size_im = size(recon_image);
    x_len = size_im(1);
    y_len = size_im(2);
    
    % Define Signal ROI in the center of the image
    signalROI = recon_image(floor(x_len/2)-N:floor(x_len/2)+N, floor(y_len/2)-N:floor(y_len/2)+N);
    
    % Define Noise ROI in the periphery of the image
    noiseROI = recon_image(1:2*N, 1:2*N);
    
    % Compute image SNR
    signal_power_recon = mean(abs(signalROI(:)).^2);
    noise_power_recon = mean(abs(noiseROI(:)).^2);
    
    % Store calculated SNR
    imageSNR(i) = 10 * log10(signal_power_recon / noise_power_recon);
    
    % Display measured SNR (fixed annotation position)
    annotation('textbox', [0.5, 0.92, 0.1, 0.05], ...
               'String', ['SNR for SNRdB = ', int2str(SNRdB(i)), ' dB: ', num2str(imageSNR(i)), ' dB'], ...
               'EdgeColor', 'none', ...
               'HorizontalAlignment', 'center', ...
               'FontSize', 12);
end

function M = fftdim(M, dim)
    for i = dim
        M   =   fftshift(fft(ifftshift(M, i), [], i), i)/sqrt(size(M,i));
    end
end

function M = ifftdim(M, dim)
    for i = dim
        M   =   fftshift(ifft(ifftshift(M, i), [], i), i)*sqrt(size(M,i));
    end
end

% function not used
function [y] = add_awgn_noise(x, SNR_dB)
    % Calculate signal power
    x(real(x) == 0) = NaN;
    x(imag(x) == 0) = NaN;
    real_signal_avg = 10 * log10(mean(abs(real(x)).^2, "all"));
    im_signal_avg = 10 * log10(mean(abs(imag(x)).^2, "all"));

    real_SNR_dB = real_signal_avg - SNR_dB;
    imag_SNR_dB = im_signal_avg - SNR_dB;
    
    % Calculate noise power in linear
    real_noise_std = sqrt(10^(real_SNR_dB / 10)) .* randn(size(x));
    im_noise_std = sqrt(10^(imag_SNR_dB / 10)) .* randn(size(x));

    complex_noise = real_noise_std + im_noise_std * 1j;
    
    % Generate Gaussian noise with appropriate power
    y = x + complex_noise;  % Complex noise

end