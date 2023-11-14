from ParallelProcessing import Scan, AzintConfig, AzimuthalIntegrator
import numpy as np

class TestScan:
    scan=Scan(name='Sample_scan_string', 
            poni_file='./test_data/calib.poni',
            mask_file='./test_data/mask.npy')
    def test_mask(self):
        assert isinstance(self.scan.mask, np.ndarray)
        assert self.scan.mask.ndim==2
        assert self.scan.mask.shape == (2167, 2070)
        assert self.scan.mask.dtype == np.dtype('uint8')
        
    
class TestAzintConfig:
    # Sample values
    sample_scan = Scan(name="Sample_scan_string", poni_file="./test_data/calib.poni", mask_file="./test_data/mask.npy")
    shape = (2167, 2070)
    pixel_size=75e-6
    radial_bins = 1000
    azimuth_bins = None
    polarization_factor = 0.95
    
    config = AzintConfig(sample=sample_scan, shape=shape, pixel_size=pixel_size, radial_bins=radial_bins,
                         azimuth_bins=azimuth_bins, polarization_factor=polarization_factor)

    def test_azint_config_creation(self):   
        assert self.config.poni_file == self.sample_scan.poni_file
        assert self.config.mask is self.sample_scan.mask
        
    def test_create_integrator(self):

        # Create AzimuthalIntegrator instance
        integrator = self.config.create_integrator()

        # Test if the integrator is created correctly
        assert isinstance(integrator, AzimuthalIntegrator)


    def test_azint_config_default_values(self):

        # Check if default values are set correctly
        assert self.config.n_splitting == 10
        assert self.config.unit == 'q'
        assert self.config.solid_angle is True
        assert self.config.error_model is None


    def test_azint_config_azimuth_bins(self):
            
        azimuth_bins_number = 360 #number azimuthal bins 
    
        config = AzintConfig(sample=self.sample_scan, shape=self.shape, pixel_size=self.pixel_size,
                            radial_bins=self.radial_bins, azimuth_bins=azimuth_bins_number,
                            polarization_factor=self.polarization_factor)
        
        assert config.azimuth_bins == azimuth_bins_number
        
        # Test when azimuth_bins is a sequence defining bin edges
        azimuth_bins_sequence = [0, 90, 180, 270, 360]  # Set as a sequence defining bin edges

        config = AzintConfig(sample=self.sample_scan, shape=self.shape, pixel_size=self.pixel_size,
                            radial_bins=self.radial_bins, azimuth_bins=azimuth_bins_sequence,
                            polarization_factor=self.polarization_factor)
       
        assert config.azimuth_bins == azimuth_bins_sequence
    
    
    
    
    
    
    