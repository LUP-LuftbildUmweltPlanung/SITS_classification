from threading import Thread
import time
from datetime import datetime

import GPUtil
import psutil

import csv



class HWMonitor(Thread):
    def __init__(self,delay,out_file_name):
        Thread.__init__(self)
        self.running = True
        self.delay = delay
        #self.optuna_trial = optuna_trial
        self.out_file_name = out_file_name
        self.outf = open(out_file_name,'w')
        self.writer = csv.writer(self.outf)
        self.write_header = True
        self.start_time = time.time()

    def run(self):
        while self.running:
            #GPUtil.showUtilization()
            #self.optuna_trial.set_user_attr("test_gpu", -99.99)
            cpu = psutil.cpu_percent(interval=None, percpu=True)
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            #disk = psutil.disk_io_counters()
            #net = psutil.net_io_counters()
            sensors = psutil.sensors_temperatures()
            fans = psutil.sensors_fans()
            GPUs = GPUtil.getGPUs()

            if self.write_header:
                header = ['datetime']
                header.append('elapsed time (s)')
                header.extend(['%Cpu'+str(i) for i in range(len(cpu))])
                header.extend(['%MEM','MEM tot','MEM used','MEM free'])
                header.extend(['%Swap','Swap tot','Swap used','Swap free'])
                for gpu in GPUs:
                    header.extend(['%GPU'+str(gpu.id),'GPU'+str(gpu.id)+' MEM Free (MB)', 'GPU'+str(gpu.id)+' MEM Used (MB)', 'GPU'+str(gpu.id)+' %MEM','GPU'+str(gpu.id)+' MEM Total (MB)'])
                for key in sensors:
                    for el in sensors[key]:
                        lbl = key+'.'+el.label+'.currentTemp'
                        header.append(lbl)

                self.writer.writerow(header)
                self.write_header = False

            data = [datetime.now()]
            data.append(round(time.time()-self.start_time))
            data.extend(cpu)
            data.extend([mem.percent,mem.total,mem.used,mem.free])
            data.extend([swap.percent,swap.total,swap.used,swap.free])
            for gpu in GPUs:
                data.extend([gpu.load,gpu.memoryFree,gpu.memoryUsed,gpu.memoryUtil*100,gpu.memoryTotal])
            for key in sensors:
                for el in sensors[key]:
                    data.append(el.current)

            self.writer.writerow(data)
            self.outf.flush()
            time.sleep(self.delay)

    def stop(self):
        self.outf.close()
        self.running = False



class PrintB(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.running = True
    def run(self):
        while self.running:
            print('B')
            time.sleep(2)
    def stop(self):
        self.running = False



if __name__ == '__main__':

    gpu_monitor = GPUMonitor(1)
    b = PrintB()

    gpu_monitor.start()
    b.start()

    time.sleep(10)
    #time.sleep(20)
    gpu_monitor.stop()
    time.sleep(10)
    b.stop()

