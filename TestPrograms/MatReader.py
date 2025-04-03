import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import netCDF4 as nc


def visualize_mat_file(file_path):
    try:
        # Load .mat file
        mat_data = sio.loadmat(file_path)

        # Create figure for MAT file analysis
        plt.figure(figsize=(15, 10))
        subplot_count = 0

        # Print out available variables
        print("MAT File Variables:")
        for key in mat_data.keys():
            if not key.startswith('__'):
                print(f"{key}: {type(mat_data[key])}")

        # Attempt to visualize specific variables
        key_priority = ['water_level', 'time_vec', 'noaa_lat', 'noaa_lon']

        for key in key_priority:
            if key in mat_data and subplot_count < 4:
                try:
                    data = mat_data[key]

                    # Skip if data is an object array or empty
                    if isinstance(data, np.ndarray) and data.dtype == object:
                        continue

                    # Ensure data is numeric
                    data = np.array(data, dtype=float)

                    subplot_count += 1
                    plt.subplot(2, 2, subplot_count)

                    # Different visualization based on data shape
                    if data.ndim == 1:
                        plt.plot(data)
                        plt.title(f'{key} (1D)')
                        plt.xlabel('Index')
                        plt.ylabel('Value')
                    elif data.ndim == 2:
                        plt.imshow(data, aspect='auto', cmap='viridis')
                        plt.colorbar()
                        plt.title(f'{key} (2D)')

                except Exception as e:
                    print(f"Error processing {key}: {e}")

        plt.tight_layout()
        plt.show()

        return mat_data

    except Exception as e:
        print(f"Comprehensive MAT file processing error: {e}")
        return None


def visualize_nc_file(file_path):
    try:
        # Open NetCDF file
        nc_data = nc.Dataset(file_path, 'r')

        # Print out all variables and their details
        print("\nNetCDF File Variables:")
        for var_name, var in nc_data.variables.items():
            print(f"{var_name}: dtype={var.dtype}, dimensions={var.dimensions}, shape={var.shape}")

        # Create figure
        plt.figure(figsize=(15, 15))
        subplot_count = 0

        # Specific target variables for visualization
        target_vars = [
            'temp', 'salinity', 'zeta',
            'u', 'v', 'uwind_speed', 'vwind_speed',
            'omega', 'ww'
        ]

        # Visualize selected variables
        for var_name in target_vars:
            if var_name not in nc_data.variables:
                continue

            var = nc_data.variables[var_name]

            # Skip non-numeric or 1D variables
            if var.ndim < 2:
                continue

            subplot_count += 1
            if subplot_count > 4:  # Limit to 4 subplots
                break

            plt.subplot(2, 2, subplot_count)

            # Handle 3D variables by taking a slice or averaging
            if var.ndim == 3:
                # For time-depth-space variables, average across time
                try:
                    data_slice = np.mean(var[:], axis=0)
                except Exception as e:
                    print(f"Error processing {var_name}: {e}")
                    continue
            else:
                data_slice = var[:]

            # Select appropriate colormap
            cmap = 'viridis'
            if 'temp' in var_name.lower():
                cmap = 'hot'
            elif 'salinity' in var_name.lower():
                cmap = 'Blues'
            elif 'wind' in var_name.lower():
                cmap = 'coolwarm'

            # Visualization
            plt.imshow(data_slice, aspect='auto', cmap=cmap)
            plt.colorbar()
            plt.title(f'{var_name} Visualization')

        plt.tight_layout()
        plt.show()

        # Detailed information about time variable
        if 'time' in nc_data.variables:
            time_var = nc_data.variables['time']
            print("\nTime Variable Details:")
            print(f"Shape: {time_var.shape}")
            print(f"Dimensions: {time_var.dimensions}")
            print(f"Units: {time_var.units if hasattr(time_var, 'units') else 'Not specified'}")

        return nc_data

    except Exception as e:
        print(f"Comprehensive NetCDF file processing error: {e}")
        return None


def main():
    # Replace with your actual file paths
    mat_file_path = '../Data/Isabel_2003.mat'
    nc_file_path = '../Data/in1_01_0001.nc'

    # Visualize .mat file
    mat_data = visualize_mat_file(mat_file_path)

    # # Visualize .nc file
    # nc_data = visualize_nc_file(nc_file_path)

    # # Close files
    # if nc_data:
    #     nc_data.close()


if __name__ == '__main__':
    main()
