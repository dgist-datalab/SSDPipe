import tensorrt as trt
import math, os, glob, time, cv2, os, sys
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from multiprocessing import Lock
from prefetch_generator import BackgroundGenerator, background
from tqdm import tqdm

class Model:

    def __init__(self, engine_path,
               max_workspace_size = 8192*1024*1024,
               fp16_mode = True,
               delay = 0):
        """
        TensorRT Engine Runner
        ----------------------
        engine_path: tensorrt engine
        fp16_mode  : if model has build with fp16 tag, than this parameter will be True
        delay      : emulate the delay (e.g., decompression overhead)
        """
        self.lock = Lock()
        self.delay = delay
        self.timecheck = 0
        self.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        print("Load Cuda Engine")
        with open(engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        print(f"Input shape : {self.engine.get_binding_shape(0)} ({self.engine.get_binding_dtype(0)})")
        print(f"Output shape: {self.engine.get_binding_shape(1)} ({self.engine.get_binding_dtype(1)})")

        self.batch_size = self.engine.get_binding_shape(1)[0]
        self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self):
        """
        Allocate host and device buffers, and create a stream.
        Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
        And also, allocate device memory for inputs and outputs.
        """
        self.h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=trt.nptype(self.engine.get_binding_dtype(0)))
        self.h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=trt.nptype(self.engine.get_binding_dtype(1)))
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        self.stream = cuda.Stream()

    @background(max_prefetch=1)
    def load_data(self, data_path, dtype=np.float32):
        """
        type size is set 2. if need to deal with float32, change this value
        """
        type_size = 2
        num_batch = os.path.getsize(data_path)//(trt.volume(self.engine.get_binding_shape(0))*type_size)
        num_images = os.path.getsize(data_path) // (trt.volume(self.engine.get_binding_shape(0))*type_size/self.batch_size)
        with open(data_path, "rb") as f:
            for batch in range(num_batch):
                self.lock.acquire()
                if batch == num_batch-1:
                    size = int(trt.volume(self.engine.get_binding_shape(0))//(self.batch_size)*(num_images % self.batch_size))
                    print(size)
                    batch_size = num_images % self.batch_size
                    np.copyto(self.h_input[:size], np.frombuffer(f.read(size*type_size), dtype=np.float16))#trt.nptype(self.engine.get_binding_dtype(0))))
                else:
                    size = trt.volume(self.engine.get_binding_shape(0))
                    batch_size = self.batch_size
                    np.copyto(self.h_input[:size], np.frombuffer(f.read(size*type_size), dtype=np.float16))#trt.nptype(self.engine.get_binding_dtype(0))))
                yield batch_size

    def do_inference(self):
        """
        Transfer input data to the GPU
        """
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        try:
            self.lock.release()
        except:
            pass
        self.context.execute_async(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle, batch_size = 1)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

    def inference(self, data_path):
        type_size = 2
        num_images = os.path.getsize(data_path) // (trt.volume(self.engine.get_binding_shape(0))*type_size/self.batch_size)
        batch_num = math.ceil(num_images / self.batch_size)
        lddt = self.load_data(data_path)

        for batch_size in tqdm(lddt, total=batch_num):
            start = time.perf_counter()
            self.do_inference()
            end = time.perf_counter()
            if self.delay*128 > (end-start):
                time.sleep(self.delay*128-(end-start))
            ee = time.perf_counter()
            self.timecheck += ee-start

if __name__ == "__main__":
    # delay?
    # To emulate the decompression overhead
    # If you write down how many seconds decompression takes per image,
    # image, it calculates itself and sleeps the rest when overlapping with inference.

    model = Model(sys.argv[1], delay=0)
    dataset = "dataset"
    num_images = os.path.getsize(dataset) // (224*224*3*2)
    print(num_images)
    start = time.perf_counter()
    model.inference(dataset)
    end   = time.perf_counter()
    print(f"inference time: {end-start}")
    print(f"inference throughput : {round(num_images/(model.timecheck),2)}IPS")






                    
                




