from threading import Thread
import time
from datetime import datetime

import GPUtil
import psutil
from psutil._common import bytes2human

import csv
import pandas as pd

class HWMonitor(Thread):
    def __init__(self,delay,out_file_name,disks):
        Thread.__init__(self)
        self.running = True
        self.delay = delay
        self.out_file_name = out_file_name
        self.outf = open(out_file_name,'w')
        self.writer = csv.writer(self.outf)
        self.write_header = True
        self.header = []
        self.start_time = time.time()
        self.disks = disks
        self.disk_usage = self.get_disk_usage()
        self.net_usage = self.get_net_usage()

        self.average = False
        self.all_data = []


    def run(self):
        while self.running:
            #GPUtil.showUtilization()
            cpu = psutil.cpu_percent(interval=None, percpu=True)
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            diff_disk_usage = self.get_diff_disk_usage(self.get_disk_usage())
            diff_net_usage = self.get_diff_net_usage(self.get_net_usage())
            #sensors = psutil.sensors_temperatures()
            fans = psutil.sensors_fans()
            GPUs = GPUtil.getGPUs()

            # Write header
            if self.write_header:
                header = ['datetime']
                header.append('elapsed time (s)')
                header.extend(['%Cpu'+str(i) for i in range(len(cpu))])
                header.extend(['%MEM','MEM tot','MEM used','MEM free'])
                header.extend(['%Swap','Swap tot','Swap used','Swap free'])

                for gpu in GPUs:
                    header.extend(['%GPU'+str(gpu.id),'GPU'+str(gpu.id)+' MEM Free (MB)', 'GPU'+str(gpu.id)+' MEM Used (MB)', 'GPU'+str(gpu.id)+' %MEM','GPU'+str(gpu.id)+' MEM Total (MB)'])

                #for key in sensors:
                #    for el in sensors[key]:
                #        lbl = key+'.'+el.label+'.currentTemp'
                #        header.append(lbl)

                for d in diff_disk_usage:
                    for key in diff_disk_usage[d]:
                        lbl = d + '.' + key
                        header.append(lbl)

                for d in diff_net_usage:
                    for key in diff_net_usage[d]:
                        lbl = d + '.' + key
                        header.append(lbl)

                self.header = header
                self.writer.writerow(header)
                self.write_header = False

            # Write data
            data = [datetime.now()]
            data.append(round(time.time()-self.start_time))
            data.extend(cpu)
            data.extend([mem.percent,mem.total,mem.used,mem.free])
            data.extend([swap.percent,swap.total,swap.used,swap.free])

            for gpu in GPUs:
                data.extend([gpu.load*100,gpu.memoryFree,gpu.memoryUsed,gpu.memoryUtil*100,gpu.memoryTotal])

            #for key in sensors:
            #    for el in sensors[key]:
            #        data.append(el.current)

            for d in diff_disk_usage:
                for key in diff_disk_usage[d]:
                    data.append(diff_disk_usage[d][key])

            for d in diff_net_usage:
                for key in diff_net_usage[d]:
                    data.append(diff_net_usage[d][key])

            if self.average:
                self.all_data.append(data)

            self.writer.writerow(data)
            self.outf.flush()
            time.sleep(self.delay)


    def stop(self):
        self.running = False
        time.sleep(2)
        self.outf.close()


    def get_disk_usage(self):
        usage = {}
        disk = psutil.disk_io_counters(perdisk=True, nowrap=True)
        for d in self.disks:
            dd = disk[d]
            usage[d] = {"read_count": dd.read_count,
                        "write_count":dd.write_count,
                        "read_bytes":dd.read_bytes,
                        "write_bytes":dd.write_bytes}
        #print(usage)
        return usage


    def get_net_usage(self):
        usage = {}
        net = psutil.net_io_counters(pernic=True, nowrap=True)
        for d in net:
            if d == 'lo':
                continue
            usage[d] = {"bytes_sent": net[d].bytes_sent,
                        "bytes_recv":net[d].bytes_recv,
                        "packets_sent":net[d].packets_sent,
                        "packets_recv":net[d].packets_recv}
        #print(usage)
        return usage


    def get_diff_disk_usage(self,usage):

        diff = {}
        for d in usage:
            diff[d] = {}
            for key in usage[d]:
                diff[d][key+'/s'] = (usage[d][key] - self.disk_usage[d][key])/self.delay

        self.disk_usage = usage

        return diff


    def get_diff_net_usage(self,usage):

        diff = {}
        for d in usage:
            diff[d] = {}
            for key in usage[d]:
                diff[d][key+'/s'] = (usage[d][key] - self.net_usage[d][key])/self.delay

        self.net_usage = usage

        return diff

    def start_averaging(self):
        self.all_data.clear()
        self.average = True

    def stop_averaging(self):
        self.average = False

    def get_averages(self):

        df = pd.DataFrame(self.all_data,columns=self.header)
        df.drop(columns=['elapsed time (s)'],inplace=True)
        df_des = df.describe(percentiles=[0.1,0.9])
        lst_des = df_des.to_dict()

        self.all_data.clear()

        return lst_des


def squeeze_hw_info(hwinfo):
    out = {}
    for key in hwinfo:
        for ink in hwinfo[key]:
            name = key+' '+ink
            val = hwinfo[key][ink]
            out[name] = val
    return out

def disk_info():

    templ = "%-17s %8s %8s %8s %5s%% %9s  %s"
    print(templ % ("Device", "Total", "Used", "Free", "Use ", "Type", "Mount"))
    for part in psutil.disk_partitions(all=False):
        if '/dev/' not in part.device:
            continue
        if '/dev/loop' in part.device:
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
            line = templ % (
                part.device.split('/')[-1],
                bytes2human(usage.total),
                bytes2human(usage.used),
                bytes2human(usage.free),
                int(usage.percent),
                part.fstype,
                part.mountpoint,
            )
        except OSError as error:
            # let's skip to next partition
            pass

        print(line)



#class PrintB(Thread):
#    def __init__(self):
#        Thread.__init__(self)
#        self.running = True
#    def run(self):
#        while self.running:
#            print('B')
#            time.sleep(2)
#    def stop(self):
#        self.running = False



if __name__ == '__main__':

    disk_info()

    #gpu_monitor = GPUMonitor(1)
    #b = PrintB()

    #gpu_monitor.start()
    #b.start()

    #time.sleep(10)
    ##time.sleep(20)
    #gpu_monitor.stop()
    #time.sleep(10)
    #b.stop()


