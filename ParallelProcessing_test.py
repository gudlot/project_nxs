from ParallelProcessing import Scan
import numpy as np

class TestScan:
    scan=Scan(name='Sample_scan_string', 
            poni_file='test_data/calib.poni',
            mask_file='test_data/mask.npy')
    def test_mask(self):
        assert isinstance(self.scan.mask, np.ndarray)
        assert self.scan.mask.ndim==2
        
    

