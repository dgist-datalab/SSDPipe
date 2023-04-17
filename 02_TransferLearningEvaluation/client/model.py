import os, paramiko
from tqdm import tqdm
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
import math
import cv2
from multiprocessing import Lock
from prefetch_generator import BackgroundGenerator, background

class Model:
    def __init__(self,
                engine_path,
                max_workspace_size = 8192*1024*1024,
                max_batch_size = 128,
                fp16_mode = True,
                in_dtype = trt.float16,
                out_dtype = trt.float16,
                evaluate_mode = False,
                classifier_path=None,
                num_classes=None,
                feature_dim = 224,
                server_dir = '/'):
        self.feature_dim = feature_dim
        self.max_batch_size = max_batch_size
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.scp_c = 0
        self.lock = Lock()
        self.evaluate_mode = evaluate_mode
        self.num_classes = num_classes
        self.offset = 0
        self.start_pos = 0
        self.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        self.server_dir = server_dir
        self.ssh = paramiko.SSHClient()
        self.ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
        self.ssh.connect('10.150.21.89', username='datai', password='datalab191066!!')
        # Build a TensorRT engine.
        print("Load Cuda Engine")
        with open(engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        print(self.engine.get_binding_shape(0), self.engine.get_binding_shape(1))
        print(self.engine.get_binding_dtype(0), self.engine.get_binding_dtype(1))
        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        self.allocate_buffers()
        # Contexts are used to perform inference.
        self.context = self.engine.create_execution_context()

#	def get_output(self):
#		return self.engine.get_binding_shape(1)

    def allocate_buffers(self):
        # Allocate host and device buffers, and create a stream.
        # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
        self.h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=trt.nptype(self.in_dtype))
        self.h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=trt.nptype(self.out_dtype))
        # Allocate device memory for inputs and outputs.
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        self.stream = cuda.Stream()

    @background(max_prefetch=3)
    def load_data(self, data_path, num_images):
        if self.in_dtype == trt.float16:
            type_size = 2
        else:
            type_size = 4
        num_batch = math.ceil(num_images / self.max_batch_size)
        #print(self.feature_dim)
        with open(data_path, "rb") as frp:
            frp.seek(self.start_pos*(self.feature_dim*self.feature_dim*3)*type_size)
            for batch in range(num_batch):
                self.lock.acquire()
                #frp.seek(self.start_pos*self.feature_dim*self.feature_dim*3*type_size)
                if batch == math.ceil((self.offset)/self.max_batch_size)-1:
                    size = self.feature_dim*self.feature_dim*3*((self.offset) % self.max_batch_size)
                    batch_size = (self.offset) % self.max_batch_size
                    np.copyto(self.h_input[:size], np.frombuffer(frp.read(self.feature_dim*self.feature_dim*3*type_size*batch_size), dtype=trt.nptype(self.in_dtype)))
                    yield batch_size
                    break
                if batch == num_batch-1:
                    size = self.feature_dim*self.feature_dim*3*(num_images % self.max_batch_size)
                    batch_size = num_images % self.max_batch_size
                    np.copyto(self.h_input[:size], np.frombuffer(frp.read(self.feature_dim*self.feature_dim*3*type_size*batch_size), dtype=trt.nptype(self.in_dtype)))
                else:
                    size = self.feature_dim*self.feature_dim*3*self.max_batch_size
                    batch_size = self.max_batch_size
                    np.copyto(self.h_input, np.frombuffer(frp.read(self.feature_dim*self.feature_dim*3*type_size*self.max_batch_size), dtype=trt.nptype(self.in_dtype)))
                #print('input:',self.h_input[:size])
                yield batch_size

    def do_inference(self):
        # Transfer input data to the GPU.
        # cuda.memcpy_htod(d_input, h_input)
        #print("cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)")
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        # Run inference.
        try:
            #print("self.lock.release()")
            self.lock.release()
        except:
            #print("pass")
            pass
        #print("self.context.execute_async(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle, batch_size = 1)")
        self.context.execute_async(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle, batch_size = 1)
        # Transfer predictions back from the GPU.
        #print("cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)")
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        # Synchronize the stream
        #print("self.stream.synchronize()")
        self.stream.synchronize()

    def do_scp(self, path):
        #self.scp_c += 1
        sftp = self.ssh.open_sftp()
        #print(path)
        #print(self.server_dir)
        #print(self.server_dir+'/'+path)
        sftp.put(path, self.server_dir+'/'+path)
        sftp.close()
        cmd = "sshpass -p 'datalab191066!!' scp "+path+" datai@10.150.21.89:"+self.server_dir
        #print_msg = path
        #if self.scp_c % 2 == 1:
        #    print("\n============= Sending feature File...=============")
        #    print("                 "+print_msg)
        #else:   
        #    print("                 "+print_msg)
        #    print("==================================================\n")
        #os.system(cmd)


    def write_label(self, source_path, final_path, offset, length, IsExtra):
        if IsExtra == True:
            mode = "ab"
        else:
            mode = "wb"

        with open(final_path, mode) as fwp:
            with open(source_path, "rb") as frp:
                frp.seek(offset*4)
                fwp.write(frp.read(length*4))

    def extract_features(self, data_path, label_path, feature_path, feature_label_path, client, append=False, IsExtra=False, extra_num=0, split=1):
        if append == False:
            mode = "wb"
        else:
            mode = "ab"
        inference = 0
        count = 0
        num_images = os.path.getsize(data_path) // (self.feature_dim*self.feature_dim*3*2)

        #for SALS loop 
        sub_loop = 0
        sub_feature = (num_images/split) - (num_images/split) % self.max_batch_size
        last_sub_feature = num_images - sub_feature*split


        if IsExtra == True:
            self.start_pos = num_images - extra_num
            num_images = extra_num


        features = np.zeros([num_images,2048], dtype=trt.nptype(self.out_dtype))
        batch_num = math.ceil(num_images / self.max_batch_size)
        tcp_num = int(batch_num*0.3)
        loop_count = 0
        throughput = 0
        balance_time = 0
        flag = 0
        sub_count = 0
        LB_mode = 0
        start = time.perf_counter()
        feature_path_list = []
        label_path_list = []
        fopen_list = []
        if IsExtra == False:
            print("Waiting for Load Balancing ...")
            for i in range(split):
                sub_feature_path = "train_feature_7_"+str(i)+".dat"
                sub_label_path = "train_label_7_"+str(i)+".dat"
                label_path_list.append(sub_label_path)
                feature_path_list.append(sub_feature_path)
                sub_f = open(sub_feature_path, mode)
                fopen_list.append(sub_f)
        else:
            print("Extra feature extraction ...")
            #pbar = tqdm(total=num_images)
            sub_feature_path = feature_path
            sub_label_path = feature_label_path
            label_path_list.append(sub_label_path)
            feature_path_list.append(sub_feature_path)
            sub_f = open(sub_feature_path, mode)
            fopen_list.append(sub_f)
        
        lddt = self.load_data(data_path, num_images)

        for batch_size in tqdm(lddt, total=batch_num):
            s = time.perf_counter()
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # ability that the image corresponds to that label
            if IsExtra == False:
                if balance_time > 30 and flag == 0:
                    flag = 1
                    #pbar = tqdm(total=num_images-count)
                    #if tcp_num == loop_count:

                    re_distribute, extra_num, extra_path, extra_label_path = self.tcp_comm(client, throughput, num_images)
                    if re_distribute == 0 and extra_num == 0:
                        self.offset = 0
                        #pbar=tqdm(total=num_images-count)
                    elif re_distribute != 0 or extra_num != 0:
                        self.offset = num_images - re_distribute
                        sub_feature = ((self.offset+extra_num)/split) - (((self.offset+extra_num)/split) % self.max_batch_size)
                        last_sub_feature = self.offset -  sub_feature*split
                        #pbar=tqdm(total=self.offset-count)
                        #print("sub_feature:", sub_feature, "last_sub_feature:", last_sub_feature)

            t1 = time.perf_counter()
            #print("**************************************\nbefore do_inference:", balance_time)
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            try:
                #print("self.lock.release()")
                self.lock.release()
            except:
                #print("pass")
                pass
            self.context.execute_async(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle, batch_size = 1)
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
            #self.do_inference()
            #print("after do_inference: ", balance_time)
            #print("**************************************")
            t2 = time.perf_counter()
            #if flag == 1 or IsExtra==True:
                #pbar.update(batch_size)
            sub_count += batch_size
            count += batch_size
                # We use the highest probability as our prediction. Its index corresponds to the predicted label.
            fopen_list[sub_loop].write(self.h_output[:batch_size*self.engine.get_binding_shape(1)[1]].tobytes())
            loop_count += 1
            inference += t2 - t1
            throughput = count/inference
            e = time.perf_counter()
            balance_time += (e-s)
            #print(sub_count, sub_feature)
            if sub_count == sub_feature and sub_loop != split-1:
                #print("bbbbbbbbbbbbbbbbb1")
                #print(sub_loop,"scp")
                self.do_scp(feature_path_list[sub_loop])
                #print("bbbbbbbbbbbbbbbb2")
                self.write_label(label_path, label_path_list[sub_loop], count-sub_count, sub_count, IsExtra)
                #print("bbbbbbbbbbbbbbbbb3")
                self.do_scp(label_path_list[sub_loop])
                #print("bbbbbbbbbbbbbbbb4")
                sub_count = 0
                sub_loop += 1


        if IsExtra == False:
            if extra_num == 0:
                #print("last scp")
                #print(count, "sub_count:", sub_count)
                #print("ssssssssssssssssssssss1")
                self.do_scp(feature_path_list[-1])
                #print("ssssssssssssssssssssss2")
                self.write_label(label_path, label_path_list[-1], count-sub_count, sub_count, IsExtra)
                #print("sssssssssssssssssssss3")
                self.do_scp(label_path_list[-1])
                #print("sssssssssssssssssssssss4")
            else:
                #print("kkkkkkkkkkkkkkkkkkkkkkkk1")
                self.write_label(label_path, label_path_list[-1], count-sub_count, sub_count, IsExtra)
                #print("kkkkkkkkkkkkkkkkkkkkkkk2")
        elif IsExtra == True:

            #print(count, "sub_count:", sub_count)
            #print("qqqqqqqqqqqqqqqqqqqqqqq1")
            self.do_scp(feature_path_list[-1])
            #print("qqqqqqqqqqqqqqqqqqqqqqq2")
            self.write_label(label_path, label_path_list[-1], count-sub_count, sub_count, IsExtra)
            #print("qqqqqqqqqqqqqqqqqqqqqqq3")
            self.do_scp(label_path_list[-1])
            #print("qqqqqqqqqqqqqqqqqqqqqq4")

        """
        #last feature
        if LB_mode == 0 and extra_num == 0:
            print("last loop scp")
            print(count, "sub_count:", sub_count)
            self.do_scp(feature_path_list[-1])
            self.write_label(label_path, label_path_list[-1], count-sub_count, sub_count, IsExtra)
            self.do_scp(label_path_list[-1])
        else if LB_mode == 0 and extra_num != 0:
            self.write_label(label_path, label_path_list[-1], count-sub_count, sub_count, IsExtra)
        """

        #pbar.close()
        end = time.perf_counter()
        total = end - start
        re_num = self.offset
        self.offset = 0
        self.start_pos = 0
        #print("Count:", count)
        #print("Inference:", inference, count/inference)
        #print("Total:", total, count/total)
        if IsExtra == False:
            return inference, total, count, extra_num, extra_path, re_num, extra_label_path, feature_path_list[-1], label_path_list[-1]
        else:
            return inference, total, count

    def tcp_comm(self, client, throughput, num_images):
        client.sendall((str(throughput)+" "+str(num_images)+" "+str(self.engine.get_binding_shape(1)[1])+" 5").encode())
        server_calc = client.recv(4096)
        calc_list = (server_calc.decode()).split()
        re_distribute = int(calc_list[0])
        extra_num = int(calc_list[1])
        extra_path = calc_list[2]
        extra_label_path = calc_list[3]
        print("\n======================LB Result======================")
        print("Re_distribute:", re_distribute)
        print("extra_num:", extra_num)
        print("extra_path:", extra_path)
        print("extra_label_path:", extra_label_path)
        print("=====================================================\n")
        return re_distribute, extra_num, extra_path, extra_label_path

    def evaluate(self, data_path, label_path):
        inference = 0
        count = 0
        top_1 = 0
        top_5 = 0
        num_images = os.path.getsize(data_path) // (self.feature_dim*self.feature_dim*3*2)

        features = np.zeros([num_images,trt.volume(self.engine.get_binding_shape(1))], dtype=trt.nptype(self.dtype))
        batch_num = math.ceil(num_images / self.max_batch_size)

        start = time.perf_counter()

        for batch_size in self.load_data(data_path, num_images):
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            t1 = time.perf_counter()
            self.do_inference()
            t2 = time.perf_counter()

            # We use the highest probability as our prediction. Its index corresponds to the predicted label.
            features[count:count+batch_size,:] = self.h_output[:batch_size*trt.volume(self.engine.get_binding_shape(1))].reshape(batch_size, trt.volume(self.engine.get_binding_shape(1)))
            count += batch_size

            inference += t2 - t1
        end = time.perf_counter()
        total = end - start

        with open(label_path, "rb") as f:
            label = np.frombuffer(f.read(), np.uint32)

        for i in range(count):
            pred = np.flip(np.argsort(features[i,:])[-5:])
            if label[i] == pred[0]:
                top_1 += 1
            if label[i] in pred:
                top_5 += 1
            # if i < 50:
            #     print(i, label[i], pred)

        print("Number of Images", count, "Top_1 accuracy:", top_1, top_1/count, "Top_5 accuracy:", top_5, top_5/count)
        print("Inference:", inference, count/inference)
        print("Total:", total, count/total)
        return inference, total, count, top_1, top_5

