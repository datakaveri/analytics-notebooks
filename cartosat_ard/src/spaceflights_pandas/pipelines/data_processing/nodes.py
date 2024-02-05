import pandas as pd
import numpy as np
import xarray as xr
import bm3d
import rasterio
from skimage import transform


def histEqualization(channel):
    """Performs Histogram Equalization on a given image channel.

    Args:
        channel (numpy.ndarray): Input image channel.

    Returns:
        numpy.ndarray: Histogram-equalized channel with enhanced contrast.

    Description:
        The histEqualization function takes a single image channel as input and applies histogram equalization
        to enhance the contrast of the channel. The process involves computing the histogram, generating the
        Cumulative Distribution Function (CDF), normalizing the CDF values, mapping the original pixel intensities
        to new values using the CDF, and reshaping the result to match the original channel's shape.

    """

    hist, bins = np.histogram(channel.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_norm = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
    channel_new = cdf_norm[channel.flatten()]
    channel_new = np.reshape(channel_new, channel.shape)
    return channel_new

def geometric_correction(original_images: pd.DataFrame) -> pd.DataFrame:

    """Performs geometric correction and histogram equalization on a collection of images.

    Args:
        original_images (pd.DataFrame): DataFrame containing information about the original images, including file paths
                                         for multispectral (mx) and panchromatic (pan) images.

    Returns:
        pd.DataFrame: DataFrame containing details of the geometrically corrected and histogram-equalized images,
                      including file names, paths to the improved mx images, original mx and pan image paths.

    Description:
        The geometric_correction function takes a DataFrame of original image information and performs geometric
        correction along with histogram equalization on the multispectral (mx) images. The process involves loading
        each mx image using xarray, extracting individual color channels (R, G, B), applying a log transform,
        normalizing pixel values to the range [0, 255], performing histogram equalization on each channel using
        the histEqualization function, and saving the resulting image. The function returns a DataFrame with
        details of the processed images.

    """

    histogram_equalized_data = pd.DataFrame(columns=["file_name", "improved_mx_file_path", "mx_file_path", "pan_file_path"])

    for index, row in original_images.iterrows():
        image_path = row["mx_file_path"]
        image_name = row["file_name"]
        mx = row["mx_file_path"]
        pan = row["pan_file_path"]

        # Open the image using xarray
        image = xr.open_dataset(image_path, engine='rasterio')

        R = image.band_data[[1]]
        G = image.band_data[[2]]
        B = image.band_data[[3]]

        log_transform = lambda x: np.log10(1 + x)

        R = log_transform(R)
        G = log_transform(G)
        B = log_transform(B)

        r_min, r_max = R.min(), R.max()
        g_min, g_max = G.min(), G.max()
        b_min, b_max = B.min(), B.max()

        r = (((R.values[0] - r_min.values) / (r_max.values-r_min.values))*255).round().astype(np.uint8)
        g = (((G.values[0] - g_min.values) / (g_max.values-g_min.values))*255).round().astype(np.uint8)
        b = (((B.values[0] - b_min.values) / (b_max.values-b_min.values))*255).round().astype(np.uint8)

        r_channel_equalized = histEqualization(r).round().astype(np.uint8)
        g_channel_equalized = histEqualization(g).round().astype(np.uint8)
        b_channel_equalized = histEqualization(b).round().astype(np.uint8)

        image = xr.open_dataset(image_path, engine='rasterio')
        image.band_data.values[1] = r_channel_equalized
        image.band_data.values[2] = g_channel_equalized
        image.band_data.values[3] = b_channel_equalized
    
        output_path = f'C:\\Users\\KOYESHA\\cartosat_ard\\data\\02_intermediate\\{image_name}_he.tif'
        image.band_data.rio.to_raster(output_path)

        histogram_equalized_data = pd.concat([histogram_equalized_data, pd.DataFrame({"file_name": f"{image_name}", "improved_mx_file_path": output_path, "mx_file_path":mx, "pan_file_path": pan}, index=[0])], ignore_index=True)

    return histogram_equalized_data

def detect_distortions(histogram_equalized_data: pd.DataFrame) -> pd.DataFrame:
    """Detects distortions from histogram-equalized images and applies noise masking on raw images.

    Args:
        histogram_equalized_data (pd.DataFrame): DataFrame containing information about histogram-equalized images,
                                                 including file paths for improved mx images and pan images.

    Returns:
        pd.DataFrame: DataFrame containing details of images with applied noise masking,
                      including file names, paths to the mx images with noise masking, and pan image paths.

    Description:
        The detect_distortions function takes a DataFrame of histogram-equalized image information and performs
        noise masking on the improved mx images. The process involves denoising the image using the BM3D algorithm,
        creating a binary mask to detect distortions, saving the noise masked raw image, and updating a new DataFrame
        with information about the processed images.

    """

    # Create an empty DataFrame to store image information
    noise_masked_data = pd.DataFrame(columns=["file_name", "mx+noise_file_path", "pan_file_path"])

    for index, row in histogram_equalized_data.iterrows():
        image_path = row["improved_mx_file_path"]
        image_name = row["file_name"]
        org_mx_path = row["mx_file_path"]
        pan_path = row["pan_file_path"]

        # Open the image using xarray
        image = xr.open_dataset(image_path, engine='rasterio')

        # Extract the data array from xarray
        image_array = image.band_data.values[1:].T.astype(np.uint8)

        # Denoise the image using bm3d
        denoised_image = bm3d.bm3d(image_array, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        denoised_image = denoised_image.clip(0, 255)

        # Create a binary mask
        target_rgb_value = np.array([255, 255, 255])

        # Create a binary mask based on the target RGB value
        binary_mask = np.zeros_like(denoised_image[:, :, 0], dtype=np.uint8)

        # Iterate through each pixel and check if it matches the target RGB value
        for i in range(denoised_image.shape[0]):
            for j in range(denoised_image.shape[1]):
                if all(denoised_image[i, j] != target_rgb_value):
                    binary_mask[i, j] = all(denoised_image[i, j])

        # Save the denoised image
        image = xr.open_dataset(org_mx_path, engine='rasterio')
        nir_tif = image.band_data.values[0]
        red_tif = image.band_data.values[1]
        green_tif = image.band_data.values[2]
        blue_tif = image.band_data.values[3]

        nir_xr = xr.DataArray(nir_tif, dims=('y', 'x'), coords={'y': image.coords['y'], 'x': image.coords['x']}, name='nir')
        red_xr = xr.DataArray(red_tif, dims=('y', 'x'), coords={'y': image.coords['y'], 'x': image.coords['x']}, name='red')
        green_xr = xr.DataArray(green_tif, dims=('y', 'x'), coords={'y': image.coords['y'], 'x': image.coords['x']}, name='green')
        blue_xr = xr.DataArray(blue_tif, dims=('y', 'x'), coords={'y': image.coords['y'], 'x': image.coords['x']}, name='blue')


        crs = image.rio.crs
        binary_mask = xr.DataArray(binary_mask, dims=('y', 'x'), coords={'y': image.coords['y'], 'x': image.coords['x']}, name='noise')
        merged_xr = xr.merge([nir_xr, red_xr, green_xr, blue_xr, binary_mask])

        merged_xr.rio.write_crs(crs, inplace=True)
        output_path = f'C:\\Users\\KOYESHA\\cartosat_ard\\data\\02_intermediate\\{image_name}_noise_masked.tif'
        merged_xr.rio.to_raster(output_path)
        

        # Append image information to the preprocess_data DataFrame
        noise_masked_data = pd.concat([noise_masked_data, pd.DataFrame({"file_name": f"{image_name}", "mx+noise_file_path": output_path, "pan_file_path": pan_path}, index=[0])], ignore_index=True)

    return noise_masked_data



def cloud_detection(noise_masked_data: pd.DataFrame) -> pd.DataFrame:
    """Detects clouds and applies cloud masking on noise-masked images.

    Args:
        noise_masked_data (pd.DataFrame): DataFrame containing information about noise-masked images,
                                           including file paths for mx images with noise masking and pan images.

    Returns:
        pd.DataFrame: DataFrame containing details of images with applied cloud masking,
                      including file names, paths to the mx images with noise and cloud masking, and pan image paths.

    Description:
        The cloud_detection function takes a DataFrame of noise-masked image information and performs cloud detection
        along with cloud masking on the mx images. The process involves applying a cloud screening algorithm based on
        thresholds, generating binary masks for cloud-free, cloudy, and potentially cloudy pixels, saving the cloud-masked
        image, and updating a new DataFrame with information about the processed images.

    """

    cloud_mask_data = pd.DataFrame(columns=["file_name", "mx+noise+cloud_file_path", "pan_file_path"])

    for index, row in noise_masked_data.iterrows():
        image_path = row["mx+noise_file_path"]
        image_name = row["file_name"]
        pan_path = row["pan_file_path"]

        # Open the image using xarray
        image = xr.open_dataset(image_path, engine='rasterio')
        image.band_data.values = image.band_data.values.round()
        image_array = image.band_data.values.T.astype(np.uint8)
        image_array = image_array.T
        image_array1 = image_array / 255.0
        # Extract the first band (Red Band)
        band_data = image_array1[0, :, :]

        # Set thresholds
        T1_threshold = 0.1
        T2_threshold = 0.15

        # Apply cloud screening algorithm
        cloud_probability = np.zeros_like(band_data)

        # Cloud-free pixels
        cloud_probability[band_data < T1_threshold] = 0.0

        # Cloudy pixels
        cloud_probability[band_data > T2_threshold] = 1.0

        # Potentially cloudy pixels
        mask_potentially_cloudy = np.logical_and(band_data >= T1_threshold, band_data <= T2_threshold)
        cloud_probability[mask_potentially_cloudy] = (band_data[mask_potentially_cloudy] - T1_threshold) / (T2_threshold - T1_threshold)


        # Define threshold values for cloud-free, cloudy, and potentially cloudy pixels
        cloud_free_threshold = 0.0
        cloudy_threshold = 1.0

        # Generate binary masks based on the thresholds
        cloud_free_mask = (cloud_probability <= cloud_free_threshold)
        cloudy_mask = (cloud_probability >= cloudy_threshold)
        potentially_cloudy_mask = np.logical_and(cloud_probability > cloud_free_threshold, cloud_probability < cloudy_threshold)
        
        # Save the cloud masked image
        image = xr.open_dataset(image_path, engine='rasterio')
        nir_tif = image.band_data.values[0]
        red_tif = image.band_data.values[1]
        green_tif = image.band_data.values[2]
        blue_tif = image.band_data.values[3]
        noise_tif = image.band_data.values[4]

        nir_xr = xr.DataArray(nir_tif, dims=('y', 'x'), coords={'y': image.coords['y'], 'x': image.coords['x']}, name='nir')
        red_xr = xr.DataArray(red_tif, dims=('y', 'x'), coords={'y': image.coords['y'], 'x': image.coords['x']}, name='red')
        green_xr = xr.DataArray(green_tif, dims=('y', 'x'), coords={'y': image.coords['y'], 'x': image.coords['x']}, name='green')
        blue_xr = xr.DataArray(blue_tif, dims=('y', 'x'), coords={'y': image.coords['y'], 'x': image.coords['x']}, name='blue')
        noise_xr = xr.DataArray(noise_tif, dims=('y', 'x'), coords={'y': image.coords['y'], 'x': image.coords['x']}, name='noise')

        crs = image.rio.crs
        potentially_cloudy_mask = xr.DataArray(potentially_cloudy_mask, dims=('y', 'x'), coords={'y': image.coords['y'], 'x': image.coords['x']}, name='cloud')
        merged_xr = xr.merge([nir_xr, red_xr, green_xr, blue_xr, noise_xr, potentially_cloudy_mask])

        merged_xr.rio.write_crs(crs, inplace=True)
        output_path = f'C:\\Users\\KOYESHA\\cartosat_ard\\data\\03_primary\\{image_name}_cloud_masked.tif'
        merged_xr.rio.to_raster(output_path)
        
        cloud_mask_data = pd.concat([cloud_mask_data, pd.DataFrame({"file_name": f"{image_name}", "mx+noise+cloud_file_path": output_path, "pan_file_path": pan_path}, index=[0])], ignore_index=True)

    return cloud_mask_data

def stretch(bands, lower_percent=2, higher_percent=98):
    """Performs contrast stretching on a set of image bands.

    Args:
        bands (numpy.ndarray): Image bands to undergo contrast stretching.
        lower_percent (int, optional): Lower percentile for intensity scaling. Defaults to 2.
        higher_percent (int, optional): Higher percentile for intensity scaling. Defaults to 98.

    Returns:
        numpy.ndarray: Image bands after applying contrast stretching.

    Description:
        The stretch function takes a set of image bands and performs contrast stretching individually on each color channel.
        Contrast stretching enhances the contrast of an image by scaling pixel intensities within a specified percentile range.
        The function iterates over the color channels, calculates the scaling parameters based on the given percentiles, and applies
        the contrast stretching transformation to each channel independently.

    """

    out = np.zeros_like(bands)
    
    for i in range(bands.shape[0]):  # Iterate over color channels
        a = 0 
        b = 255 
        c = np.percentile(bands[i], lower_percent)
        d = np.percentile(bands[i], higher_percent) 
        t = np.zeros_like(bands[i], dtype=np.float32)
        valid_mask = (d - c) != 0  # Avoid division by zero
        t[valid_mask] = a + (bands[i][valid_mask] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[i] = t.astype(np.uint8)
    
    return out.astype(np.uint8)

def pansharpening (cloud_mask_data: pd.DataFrame) -> pd.DataFrame:
    """Performs pansharpening on images with cloud masking.

    Args:
        cloud_mask_data (pd.DataFrame): DataFrame containing information about cloud-masked images,
                                         including file paths for mx images with noise and cloud masking, and pan images.

    Returns:
        pd.DataFrame: DataFrame containing details of pansharpened images,
                      including file names, paths to the mx images with noise, cloud, and pansharpened masking, and pan image paths.

    Description:
        The pansharpening function takes a DataFrame of cloud-masked image information and performs pansharpening
        on the mx images. The process involves rescaling the multispectral (mx) bands, combining them with the panchromatic
        (pan) band, and stretching the resulting image. The function saves the pansharpened image and updates a new DataFrame
        with information about the processed images.

    """

    pansharpened_data = pd.DataFrame(columns=["file_name", "mx+noise+cloud_file path", "pan_file_path", "pansharpened_file_path"])

    for index, row in cloud_mask_data.iterrows():
        mx_path = row["mx+noise+cloud_file_path"]
        pan_path = row["pan_file_path"]
        image_name = row["file_name"]

        mx_image = xr.open_dataset(mx_path, engine='rasterio')
        mx_image = mx_image.band_data.values.T
        mx_image = mx_image.T

        pan_image = xr.open_dataset(pan_path, engine='rasterio')
        pan_image = pan_image.band_data.values.T
        pan_image = pan_image.squeeze()

        desired_shape = (1002, 1002)

        if pan_image.shape != desired_shape:
            print(image_name)
            # Create a new array filled with zeros
            filled_pan_data = np.zeros(desired_shape, dtype=pan_image.dtype)

            # Copy the existing data into the new array
            filled_pan_data[:pan_image.shape[0], :pan_image.shape[1]] = pan_image

            # Update pan_data with the filled array
            pan_image = filled_pan_data

        # get m_bands
        rgbn = np.empty((mx_image.shape[0], mx_image.shape[1], mx_image.shape[2]))
        rgbn[0, :, :] = mx_image[1, :, :]  # red
        rgbn[1, :, :] = mx_image[2, :, :]  # green
        rgbn[2, :, :] = mx_image[3, :, :]  # blue
        rgbn[3, :, :] = mx_image[0, :, :]  # NIR-1
    

        # scaled them    
        rgbn_scaled = transform.rescale(rgbn, 3.9140625)

        # Convert to NumPy arrays
        R = np.array(rgbn_scaled[1, :, :])
        G = np.array(rgbn_scaled[2, :, :])
        B = np.array(rgbn_scaled[3, :, :])
        I = np.array(rgbn_scaled[0, :, :])
        

        sharpened = None
        all_in = R + G + B + I
        print("all_in shape:", all_in.shape)
        print("pan_image shape:", pan_image.shape)

        prod = np.multiply(all_in, pan_image)
        
        r = (R/prod)[:, :, np.newaxis]
        g = (G/prod)[:, :, np.newaxis]
        b = (B/prod)[:, :, np.newaxis]
        i = (I/prod)[:, :, np.newaxis]
        
        sharpened = np.concatenate([r, g, b, i], axis=2)
        sharpened = np.transpose(sharpened, (2, 0, 1))
        stretched_image = stretch(sharpened)

        image = xr.open_dataset(pan_path, engine='rasterio')

        y_coord = np.linspace(image.coords['y'][0], image.coords['y'][-1], stretched_image.shape[1])
        x_coord = np.linspace(image.coords['x'][0], image.coords['x'][-1], stretched_image.shape[2])

        r = xr.DataArray(stretched_image[0,:,:], dims=('y', 'x'), coords={'y': y_coord, 'x': x_coord}, name='r')
        g = xr.DataArray(stretched_image[1,:,:], dims=('y', 'x'), coords={'y': y_coord, 'x': x_coord}, name='g')
        b = xr.DataArray(stretched_image[2,:,:], dims=('y', 'x'), coords={'y': y_coord, 'x': x_coord}, name='b')
        n = xr.DataArray(stretched_image[3,:,:], dims=('y', 'x'), coords={'y': y_coord, 'x': x_coord}, name='n')
        
        crs = image.rio.crs
        new = xr.merge([r, g, b, n])
        new.rio.write_crs(crs, inplace=True)
        output_path = f'C:\\Users\\KOYESHA\\cartosat_ard\\data\\04_feature\\{image_name}_pansharpened.tif'
        new.rio.to_raster(output_path)

        pansharpened_data = pd.concat([pansharpened_data, pd.DataFrame({"file_name": f"{image_name}", "mx+noise+cloud_file path": mx_path, "pan_file_path": pan_path, "pansharpened_file_path": output_path}, index=[0])], ignore_index=True)

    return pansharpened_data




    


