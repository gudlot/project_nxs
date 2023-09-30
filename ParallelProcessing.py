import numpy as np
import h5py
from typing import Optional, Union, Sequence


##### Zdenek's injection to open h5 file, it has to be before h5py
import os
import bitshuffle # optional, check code below it is never needed
#import bitshuffle.h5 # optional,if you know filter number it is never needed

# this is manipulating environment (only needed if plugin path not properly set)
plugin_dir = os.path.join(os.path.dirname(bitshuffle.__file__),'plugin')
os.environ["HDF5_PLUGIN_PATH"] = plugin_dir

os.system("echo 'Test if the HDF5_PLUGIN_PATH is set:' $(env | grep HDF5)")
##### 


from tqdm import tqdm
from multiprocessing import Process
import zmq
import uuid
from azint import AzimuthalIntegrator
#import hdf5plugin #not necessary with Zdenek's injection above

from dataclasses import dataclass, InitVar

import hdf5plugin



@dataclass
class Scan:
    """Basic class for keeping track of a scan."""
    name: str #scan_name 
    poni_file: Union[str, os.PathLike] #path/to/poni_file or example.poni
    mask_file: Union[str, os.PathLike] #path/to/mask_file or example.npy 
        
    def __post_init__(self):
        self.mask=np.load(self.mask_file)
        #how to make an np.array immutable
        self.mask.flags.writeable=False
        # alternative
        #self.mask.setflags(write=False)
        #be careful, if array is sliced, it is suddenly again mutable
            
  
    
@dataclass
class AzintConfig:
    """"Class to configure azint"""
    sample: InitVar[Scan]
    shape: tuple[int, int]
    pixel_size: float
    radial_bins: Union[int, Sequence]
    azimuth_bins: Optional[Union[int, Sequence, None]]
    polarization_factor: float
    n_splitting: int=10
    unit: str='q'
    solid_angle: bool = True
    error_model: Optional[str]=None

    def __post_init__(self, sample):
        self.poni_file = sample.poni_file
        self.mask= sample.mask
 
    def create_integrator(self) -> AzimuthalIntegrator:
        return AzimuthalIntegrator(self.poni_file, self.shape, self.pixel_size, self.n_splitting, self.radial_bins, self.azimuth_bins,  self.unit, self.mask,
                                   self.solid_angle, self.polarization_factor, self.error_model)

class DataReader():
    """This class extracts datasets from hdf5 and unifies them for the DataWriter. 
       If there is one day I1 besides I0, it could be necessary to perform calculations here as well
       Brings the data into the required format for the writer
    """
    def __init__(self, scan: Scan) -> None:
        raise NotImplementedError()
        
    def read_xrd_dset(self):
        """returns a generator producing images as ndarray"""
        raise NotImplementedError()
    
    def read_i0_dset(self):
        """ return i0 values for normalisation """
        raise NotImplementedError()
    
    def read_shape_dset(self):
        """ return shape of scan """
        raise NotImplementedError()
    
    def read_dims_dset(self):
        """ return axis names """
        raise NotImplementedError()
    
    def read_sample_dset(self):
        """ return sample name (given by user during measurement) """
        raise NotImplementedError()
    
    def read_scan_command_dset(self):
        """ scan command """
        raise NotImplementedError()

    def read_poni_file(self):
        """ read poni file """
        raise NotImplementedError()


class ESRFDataReader(DataReader):
    """ ESRFDataReader based on DataReader for ESRF data files """
    def __init__(self, fname: Union[str, os.PathLike],  scan: Scan) -> None:
        self._fname=fname
        self._scan=scan
        self._nx_instrument=f'{scan.name}/instrument'
        self._fh = None 
      
    def _h5_dataset(self, dset_path:str):
        if not hasattr(self, '_fh'):
            raise RuntimeError(f'Please use context manager.')
        return self._fh[dset_path]
      
    def read_xrd_dset(self):
        return self._h5_dataset(f'{self._nx_instrument}/eiger/data')
           
    def read_i0_dset(self):
        return self._h5_dataset(f'{self._nx_instrument}/ct34/data')
          
    
    def read_shape_dset(self):
        """ shape, [dim0, dim1] 
            slow axis, fast axis
        """
        technique = self._h5_dataset(f'{self._scan.name}/technique')
        return [technique[t][()] for t in technique]
    
    def read_dims_dset(self):
        """ [dim0, dims1]; names of slow axis, fast axis  as list
        """
        axis = self._h5_dataset(f'{self._scan.name}/technique')
        return [t for t in axis]
    
    def read_sample_dset(self): 
        """ sample name (given by user during measurement) """
        return self._h5_dataset(f'{self._scan.name}/sample/name')
    
    def read_scan_command_dset(self):
        """ scan command """
        return self._h5_dataset(f'{self._scan.name}/title')
   
    def __enter__(self): 
        self._fh= h5py.File(self._fname, 'r')
        return self
       
    def __exit__(self, exc_type, exc_value, exc_tb):
        print('Closing file')
        self._fh.close()
        

class DataWriter():
    """ Template class for a DataWriter
        Usage: With context manager
    
        What data do I need? What data/metadata is relevant to store? 
        #My first
        - Information from the Scan
            - name (scan name)
            - mask
            - poni
        #Metadata
        - sample name
        - title (contains the scan name) 
        - I0 (ESRF ct34)
        - shape of scan (at ESRF technique)
        #XRDdata
        - Converted XRD data (projection, cake, radial_axis (q), azimuth_axis ('phi'))
        #XRFdata (later)
        - Extendible: entry for XRF data (linked? )
        How to assemble the path to the output folder? 
        - output_path
        - name (scan name)
        - suffix        
    """
    
    def __init__(self, fname,  output_path: Union[str, os.PathLike], fsuffix:str, scan: Scan, datareader: DataReader) -> None:
        self._fname=fname
        self._output_path= output_path 
        self._scan = scan
        self._datareader= datareader
        self._fsuffix = fsuffix
        self._fh=None
        self._output_abs_fname=None
        
    def _file_checks(self) -> None:
        if not os.path.exists(self._output_path):
            raise FileNotFoundError(f'{self._output_path} does not exist.')      
        
        #output_file_path = os.path.join(self._output_path, f'{self._scan.name}_{self._fsuffix}.h5')
        output_file_path = os.path.join(self._output_path, f'{os.path.splitext(os.path.basename(self._fname))[0]}_{self._scan.name}_{self._fsuffix}.h5')
        if os.path.isfile(output_file_path):
            raise FileExistsError(f'{output_file_path} already exists.')      
        
    def _initialize_file(self):
        self._fh = h5py.File(self._output_abs_fname, 'w')
        print(f'Writing empty h5-file: {self._output_abs_fname}')    
           
    def __enter__(self): 
        self._file_checks()
        self._output_fname= f'{os.path.splitext(os.path.basename(self._fname))[0]}_{self._scan.name}_{self._fsuffix}.h5'
        self._output_abs_fname= os.path.join(self._output_path, self._output_fname)
        
        self._initialize_file()
                      
        #copy data from raw file into new file
        self._write_i0_dset()
        self._write_shape_dset()
        self._write_dims_dset()
        self._write_sample_dset()
        self._write_scan_command_dset()
        self._write_mask()
        self._write_poni_name()
         
        return self 
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        print('Closing file')
        self._fh.close()
    
    #decorator function (for public functions)
    def needs_context_manager(f):
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, '_fh'):
                raise RuntimeError(f'Please use context manager.')
            return f(self, *args, **kwargs)
        return wrapper
       
    def _write_i0_dset(self):
        """ i0 signal """
        i0_dset= self._datareader.read_i0_dset()
        self._fh.create_dataset('i0', data=i0_dset)
        
    def _write_shape_dset(self):
        """ shape """
        shape_dset = self._datareader.read_shape_dset()
        self._fh.create_dataset('shape', data=shape_dset)
        
    def _write_dims_dset(self):
        """ axis """
        dims_dset =  self._datareader.read_dims_dset()
        self._fh.create_dataset('dims', data=dims_dset) 
        
    def _write_sample_dset(self):
        """ sample name (given by user during measurement) """
        sample_dset =  self._datareader.read_sample_dset()
        self._fh.create_dataset('sample', data=sample_dset)   
        
    def _write_scan_command_dset(self):
        """" scan command """
        scan_command_dset =  self._datareader.read_scan_command_dset()
        self._fh.create_dataset('scan', data=scan_command_dset) 
        
    def _write_poni_name(self):
        self._fh.create_dataset('poni_file', data=os.path.basename(self._scan.poni_file))   
    
    def _write_mask(self):
        dset=self._fh.create_dataset('mask', data=self._scan.mask)
        dset.attrs['name']= os.path.basename(self._scan.mask_file)
        
    #public function accessible via Processing
    @needs_context_manager    
    def write_radial_dset(self, radial_unit, radial_axis):
        """ q axis 
        radial_unit: str ['q' or 'q_nm']
        radial_axis: np.ndarray
        """
        self._fh.create_dataset(f'{radial_unit}', data=radial_axis)

        pass
    
    #public function accessible via Processing
    @needs_context_manager    
    def write_azimuth_dset(self, azimuth_axis):
        """ phi, azimuth axis
            azimuth_axis: np.ndarray
        """
        self._fh.create_dataset(f'phi', data=azimuth_axis)
    
    @needs_context_manager 
    def write_xrd_cake_dset(self, img_number, cake_data):
        dset=self._fh.get('cake')
        if not dset:
            #Only created once
            images = self._datareader.read_xrd_dset()
            dset=self._fh.create_dataset('cake', shape=(len(images), *cake_data.shape), dtype=cake_data.dtype)
        dset[img_number]=cake_data
    
    @needs_context_manager 
    def write_xrd_projection_dset(self, img_number, projection_data):
        dset=self._fh.get('projection')
        if not dset:
            #Only created once
            images = self._datareader.read_xrd_dset()
            dset=self._fh.create_dataset('projection', shape=(len(images), *projection_data.shape), dtype=projection_data.dtype)
        
        dset[img_number]=projection_data
   
@dataclass
class IntegratorResult: 
    """Collection of integrated results"""
    projection: np.ndarray
    cake: np.ndarray
    radial_axis: np.ndarray
    azimuth_axis: Optional[np.ndarray]=None


class Integrator:
     #Do I need here a constuctor? No  

    def radial_unit(self):
        """ define radial unit based on availability in integrator library """
        raise NotImplementedError()
    
    def calculate(self, img : np.ndarray) -> IntegratorResult:
        """ 
        input: img, 1 single image
        output: IntegratorResult(projection, cake, radial_axis, azimuth_axis)    
        """
        raise NotImplementedError()

 
class PyFaiIntegrator(Integrator):
    """ Template for integrator class based on pyFAI """
    def __init__(self, npt_rad):
            #here you have to hide the optional parameters for pyfai.integrate2d (all potential)
            #npt_rad is a necessaity
            self._npt_rad=npt.rad
            pass
    def calculate(self, img : np.ndarray) -> IntegratorResult:
        #ai.integrate2d(img, self._npt_rad)
        raise NotImplementedError()
 
    
class AzIntIntegrator(Integrator):
    """ Integrator class based on azint """
    def __init__(self, config: AzintConfig): 
        self._ai = config.create_integrator()
        
    def radial_unit(self):
        return self._ai.unit
    
    def calculate(self, img : np.ndarray) -> IntegratorResult:
        """  Calculate cake and projection for a single image.
        Args:
            img (np.ndarray): Input image.
        Returns:
            IntegratorResult: Result of the integration.
        """
        
        # add additional mask in case Eiger had problems on ID13 (dynamic masking)
        #TODO: make it optional later 
        mask = np.zeros(img.shape, dtype=np.uint8)
        #mask[img == 2**32-1] = 1
        mask[img == np.iinfo(img.dtype).max] = 1 #np.iinfo(np.uint32).max = 2**32-1 or more general img.dtype
                         
        signal, _, norm = self._ai.integrate(img, mask=mask,normalized=False) 
        
        # projection onto the q axis
        projection = self._save_divide(np.sum(signal, axis=0), np.sum(norm, axis=0))
        cake = self._save_divide(signal, norm)
        
        return IntegratorResult(projection, cake, self._ai.radial_axis, self._ai.azimuth_axis)
    
    def _save_divide(self, a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0.0)            

class SingleTestProcessing:
    
    def __init__(self, nworkers,  integrator: Integrator, datareader: DataReader, datawriter: DataWriter):
        self._integrator=integrator
        self._datareader=datareader
        self._nworkers=nworkers 
        self._datawriter=datawriter 
    
    def execute_processing(self):
        """ Main method to start and execute the data processing. 
        """
        images = self._datareader.read_xrd_dset() 
                
        first_res = self._integrator.calculate(images[0])
        self._datawriter.write_radial_dset(self._integrator.radial_unit(), first_res.radial_axis)
        self._datawriter.write_azimuth_dset(first_res.azimuth_axis)


        nimages = len(images)
        nimages = 2000 #override for test purpose

        pbar = tqdm(total=nimages)
        for img_number in range(nimages):
            img = images[img_number]

            #conversion of raw detector images to something useful
            result = self._integrator.calculate(img)

            self._datawriter.write_xrd_cake_dset(img_number=img_number, cake_data=result.cake)
            self._datawriter.write_xrd_projection_dset(img_number=img_number,projection_data=result.projection)
            pbar.update()
        pbar.close()
                
class ParallelProcessing:
    """ Class to distribute the processing of the conversion of the detector images
        Based on zmq. It has 3 private methods worker, collector, ordered_recv.
        
        integrator: an integrator object (Azint, pyFAI)
        dataread: functionality to read hdf5 files
    
    """
        
    def __init__(self, nworkers,  integrator: Integrator, datareader: DataReader, datawriter: DataWriter):
        self._integrator=integrator
        self._datareader=datareader
        self._nworkers=nworkers 
        self._datawriter=datawriter 
        # unique socket address
        self._zmq_socket_addr=f'ipc:///tmp/{uuid.uuid4()}'


        print(self._datareader._fname)
        print(self._integrator._ai)
        
    
    def _worker(self, worker_id):
        context = zmq.Context()
        push_sock = context.socket(zmq.PUSH)
        push_sock.connect(self._zmq_socket_addr)
        
        #load 3d np.array with 2d images (xrd data)
        images = self._datareader.read_xrd_dset() 
                
        nimages = len(images)
        for i in range(worker_id, nimages, self._nworkers):
            img = images[i]

            #conversion of raw detector images to something useful
            result = self._integrator.calculate(img)
            
            #push out: a tuple
            push_sock.send_pyobj((i, result))
    
    # Unordered_recv can be used if order does not matter
    def _unordered_recv(self, sock):
        while True:
            img_number, result = sock.recv_pyobj()
            yield (img_number, result)
    
    # Ordered_recv can be used for live data analysis or 
    # when order matters
    # This function requires more memory because you need to keep data in memory
    def _ordered_recv(self, sock):
        cache = {}
        next_img_number = 0
        while True:
            current_img_number, current_result = sock.recv_pyobj()
            if current_img_number == next_img_number:
                yield (current_img_number, current_result)
                next_img_number += 1
                while next_img_number in cache:
                    next_result = cache.pop(next_img_number)
                    yield (next_img_number, next_result)
                    next_img_number += 1
            else:
                cache[current_img_number] = current_result    
   
    
    def _collector(self):
        print('collector')
        context = zmq.Context()
        pull_sock = context.socket(zmq.PULL)
        pull_sock.bind(self._zmq_socket_addr)   
        
        #load 3d np.array with 2d images (xrd data)
        images = self._datareader.read_xrd_dset() 
        print(f'Length of images: {len(images)}')
             
        #write axes values to output h5, for this run calculate on the first image
        first_res = self._integrator.calculate(images[0])
        self._datawriter.write_radial_dset(self._integrator.radial_unit(), first_res.radial_axis)
        self._datawriter.write_azimuth_dset(first_res.azimuth_axis)
    
        pbar = tqdm(total=len(images))
        
        # You need to decide what is better for your application case
        #generator = self._unordered_recv(pull_sock)   
        generator = self._ordered_recv(pull_sock)
        
        for i in range(len(images)):
            img_number, result = next(generator)
            self._datawriter.write_xrd_cake_dset(img_number=img_number, cake_data=result.cake)
            self._datawriter.write_xrd_projection_dset(img_number=img_number,projection_data=result.projection)
            pbar.update()
            
        pbar.close()
            
      
     
    def execute_processing(self):
        """ Main method to start and execute the data processing. 
        """
        procs = []
        for i in range(self._nworkers):
            p = Process(target=self._worker, args=(i, ))
            #print(f'worker id {i}\n')
            p.start()
            procs.append(p)
        
        #self._collector(self._ai, fname, scan)
        self._collector()
        for i in range(self._nworkers):
            procs[i].join()
        
"""
     
def main():
    nworkers = 20

    # create objects     
    esrfscan=Scan(name='MIH_B_MIH_B_full_1.1', 
            poni_file='/mxn/visitors/gudrunl/mih-data-analysis/md1285/gudrun_processing/calib.poni',
            mask_file='/mxn/visitors/gudrunl/mih-data-analysis/md1285/gudrun_processing/MIH_B_MIH_B_full_mask_1.npy')
    fname='/data/visitors/nanomax/20220124/md1285/id13/md1285_id13.h5'

    scan_config=AzintConfig(sample=esrfscan, shape=(2167, 2070), pixel_size=75e-6, n_splitting=10, radial_bins=1000, azimuth_bins=360, unit='q',
                            polarization_factor=0.95)
    
    output_path='/data/visitors/nanomax/20220124/md1285/id13/XXusers'
    fsuffix='ordered_recv_fixed'
        
    with ESRFDataReader(fname, esrfscan) as esrfdata:
        integrator=AzIntIntegrator(scan_config)
        with DataWriter(fname, output_path, fsuffix, esrfscan, esrfdata) as datawriter:
            #p=SingleTestProcessing(nworkers,integrator, esrfdata, datawriter) 
            p=ParallelProcessing(nworkers,integrator, esrfdata, datawriter) 
            p.execute_processing()

"""   

if __name__ == '__main__':  
    main()