from threading import Thread
import time

import GPUtil
import psutil



class HWMonitor(Thread):
    def __init__(self,delay,optuna_trial):
        Thread.__init__(self)
        self.running = True
        self.delay = delay
        self.optuna_trial = optuna_trial

    def run(self):
        while self.running:
            #GPUtil.showUtilization()
            #self.optuna_trial.set_user_attr("test_gpu", -99.99)
            cpu = psutil.cpu_times_percent(interval=None, percpu=True)
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            disk = psutil.disk_io_counters()
            net = psutil.net_io_counters()
            sensors = psutil.sensors_temperatures()
            fans = psutil.sensors_fans()
            GPUs = GPUtil.getGPUs()
            for gpu in GPUs:
                print("=============================")
                print("GPU " + str(gpu.id)  + " | RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.id,gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

            print("=============================")
            print("CPU")
            print(cpu)
            print("=============================")
            print("MEM")
            print(mem)
            print("=============================")
            print("SWAP")
            print(swap)
            print("=============================")
            print("DISK")
            print(disk)
            print("=============================")
            print("NET")
            print(net)
            print("=============================")
            print("SENSORS")
            print(sensors)
            print("=============================")
            print("FANS")
            print(fans)
            print("=============================")
            print()
            time.sleep(self.delay)
    def stop(self):
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

