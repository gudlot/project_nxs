from ParallelProcessing import Scan, AzintConfig, AzimuthalIntegrator
import numpy as np

class Sample_scan_string:
    scan=Scan(name='Sample_scan_string', 
            poni_file='./test_data/calib.poni',
            mask_file='./test_data/mask.npy')
    def test_mask(self):
        assert isinstance(self.scan.mask, np.ndarray)
        assert self.scan.mask.ndim==2
        assert self.scan.mask.shape == (2167, 2070)
        assert self.scan.mask.dtype == np.dtype('uint8')
        
    

def test_azint_config_creation():
    # Sample values
    sample_scan = Scan(name="Sample_scan_string", poni_file="./test_data/calib.poni", mask_file="./test_data/mask.npy")
    shape = (2167, 2070)
    pixel_size=75e-6
    radial_bins = 1000
    azimuth_bins = None
    polarization_factor = 0.95

    # Create AzintConfig instance
    config = AzintConfig(sample=sample_scan, shape=shape, pixel_size=pixel_size, radial_bins=radial_bins,
                         azimuth_bins=azimuth_bins, polarization_factor=polarization_factor)

    # Test if attributes are set correctly
    assert config.shape == shape
    assert config.pixel_size == pixel_size
    assert config.radial_bins == radial_bins
    assert config.azimuth_bins == azimuth_bins
    assert config.polarization_factor == polarization_factor
    assert isinstance(config.n_splitting, int)
    assert config.unit == 'q'
    assert config.solid_angle is True
    assert config.error_model is None

    # Test if __post_init__ sets additional attributes
    assert config.poni_file == sample_scan.poni_file
    assert config.mask is sample_scan.mask
    
    
def test_create_integrator():
    # Sample values
    sample_scan = Scan(name="Sample_scan_string", poni_file="./test_data/calib.poni", mask_file="./test_data/mask.npy")
    shape = (2167, 2070)
    pixel_size=75e-6
    radial_bins = 1000
    azimuth_bins = None
    polarization_factor = 0.95

    # Create AzintConfig instance
    config = AzintConfig(sample=sample_scan, shape=shape, pixel_size=pixel_size, radial_bins=radial_bins,
                         azimuth_bins=azimuth_bins, polarization_factor=polarization_factor)

    # Create AzimuthalIntegrator instance
    integrator = config.create_integrator()

    # Test if the integrator is created correctly
    assert isinstance(integrator, AzimuthalIntegrator)


def test_azint_config_default_values():
    # Test default values for optional parameters
    # Sample values
    sample_scan = Scan(name="Sample_scan_string", poni_file="./test_data/calib.poni", mask_file="./test_data/mask.npy")
    shape = (2167, 2070)
    pixel_size=75e-6
    radial_bins = 1000
    azimuth_bins = None
    polarization_factor = 0.95

    # Create AzintConfig instance without providing optional parameters
    config = AzintConfig(sample=sample_scan, shape=shape, pixel_size=pixel_size, radial_bins=radial_bins,
                         azimuth_bins=azimuth_bins, polarization_factor=polarization_factor)

    # Check if default values are set correctly
    assert config.n_splitting == 10
    assert config.unit == 'q'
    assert config.solid_angle is True
    assert config.error_model is None


def test_azint_config_azimuth_bins():
    # Sample values
    sample_scan = Scan(name="Sample_scan_string", poni_file="./test_data/calib.poni", mask_file="./test_data/mask.npy")
    shape = (2167, 2070)
    pixel_size=75e-6
    radial_bins = 1000 
    azimuth_bins_number = 360 #number azimuthal bins 
    polarization_factor = 0.95
    
    config = AzintConfig(sample=sample_scan, shape=shape, pixel_size=pixel_size,
                         radial_bins=radial_bins, azimuth_bins=azimuth_bins_number,
                         polarization_factor=polarization_factor)
    
    assert config.azimuth_bins == azimuth_bins_number
    
     # Test when azimuth_bins is a sequence defining bin edges
    azimuth_bins_sequence = [0, 90, 180, 270, 360]  # Set as a sequence defining bin edges

    config = AzintConfig(sample=sample_scan, shape=shape, pixel_size=pixel_size,
                         radial_bins=radial_bins, azimuth_bins=azimuth_bins_sequence,
                         polarization_factor=polarization_factor)

    assert config.azimuth_bins == azimuth_bins_sequence
    
    
    
    
    
    
    