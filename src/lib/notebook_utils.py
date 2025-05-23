import pkg_resources
import types
import psutil
import platform
from datetime import datetime
import GPUtil
from tabulate import tabulate
import sys
import os
import pip #needed to use the pip functions

def print_python_info():
    print(f"Current Python executable: {sys.executable}\t{platform.python_version()} {sys.version}")
    print(f"Current Directory: {os.getcwd()}")

def print_imports(globalvals: dict):
    """Hack to print the packages used in the notebook, in the style used for requirements.txt
    From https://stackoverflow.com/questions/40428931/package-for-listing-version-of-packages-used-in-a-jupyter-notebook

    Args:
            globalvals (dict): call globals() from the context you want the packages from. Aka within the notebook
    """

    def get_imports():
        for name, val in globalvals.items():
            if isinstance(val, types.ModuleType):
                # Split ensures you get root package,
                # not just imported function
                name = val.__name__.split(".")[0]
            elif isinstance(val, type):
                name = val.__module__.split(".")[0]
            # Some packages are weird and have different
            # imported names vs. system/pip names. Unfortunately,
            # there is no systematic way to get pip names from
            # a package's imported name. You'll have to add
            # exceptions to this list manually!
            poorly_named_packages = {
                "PIL": "Pillow",
                "sklearn": "scikit-learn",
                "GPUtil": "gputil",
            }
            if name in poorly_named_packages.keys():
                name = poorly_named_packages[name]
            yield name.lower()
    imports = list(set(get_imports()))

    # The only way I found to get the version of the root package
    # from only the name of the package is to cross-check the names
    # of installed packages vs. imported packages
    modules = []
    for m in sys.builtin_module_names:
        if m.lower() in imports and m !='builtins':
            modules.append((m,'Python BuiltIn'))
            imports.remove(m.lower())

    for m in pkg_resources.working_set:
        if m.project_name.lower() in imports and m.project_name!="pip":
            modules.append((m.project_name, m.version))
            imports.remove(m.project_name.lower())

    for m in sys.modules:
        if m.lower() in imports and m !='builtins':
            modules.append((m,'unknown'))

    # print('System=='+platform.system()+' '+platform.release()+'; Version=='+platform.version())
    print("="*20, "Imported Packages", "="*20)
    for r in modules:
        print("{}=={}".format(*r))

def get_size(bytes):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}B"
        bytes /= factor

def print_machine_info():
    """From https://thepythoncode.com/article/get-hardware-system-information-python
    """
    print("="*40, "System", "="*40)
    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Node Name: {uname.node}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")

    # let's print CPU information
    print("="*40, "CPU", "="*40)
    # number of cores
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
    # CPU usage
    print("CPU Usage Per Core:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        print(f"Core {i}: {percentage}%")
    print(f"Total CPU Usage: {psutil.cpu_percent()}%")

    # Memory Information
    print("="*40, "Memory", "="*40)
    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)}")
    print(f"Percentage: {svmem.percent}%")
    print("="*20, "SWAP", "="*20)
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    print(f"Total: {get_size(swap.total)}")
    print(f"Free: {get_size(swap.free)}")
    print(f"Used: {get_size(swap.used)}")
    print(f"Percentage: {swap.percent}%")

    # Disk Information
    print("="*40, "Disk", "="*40)
    print("Partitions and Usage:")
    # get all disk partitions
    partitions = psutil.disk_partitions()
    for partition in partitions:
        print(f"=== Device: {partition.device} ===")
        print(f"  Mountpoint: {partition.mountpoint}")
        print(f"  File system type: {partition.fstype}")
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            # this can be catched due to the disk that
            # isn't ready
            continue
        print(f"  Total Size: {get_size(partition_usage.total)}")
        print(f"  Used: {get_size(partition_usage.used)}")
        print(f"  Free: {get_size(partition_usage.free)}")
        print(f"  Percentage: {partition_usage.percent}%")
    # get IO statistics since boot
    disk_io = psutil.disk_io_counters()
    print(f"Total read: {get_size(disk_io.read_bytes)}")
    print(f"Total write: {get_size(disk_io.write_bytes)}")

    # Network information
    print("="*40, "Network", "="*40)
    # get all network interfaces (virtual and physical)
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            print(f"=== Interface: {interface_name} ===")
            if str(address.family) == 'AddressFamily.AF_INET':
                print(f"  IP Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == 'AddressFamily.AF_PACKET':
                print(f"  MAC Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast MAC: {address.broadcast}")
    # get IO statistics since boot
    net_io = psutil.net_io_counters()
    print(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    print(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")

    # GPU information
    print("="*40, "GPU", "="*40)
    gpus = GPUtil.getGPUs()
    list_gpus = []
    for gpu in gpus:
        gpu_id = gpu.id
        gpu_name = gpu.name
        gpu_load = f"{gpu.load*100}%" # get % percentage of GPU usage of that GPU
        gpu_free_memory = f"{gpu.memoryFree}MB"
        gpu_used_memory = f"{gpu.memoryUsed}MB"
        gpu_total_memory = f"{gpu.memoryTotal}MB"
        gpu_temperature = f"{gpu.temperature} °C"
        gpu_uuid = gpu.uuid
        list_gpus.append((
            gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
            gpu_total_memory, gpu_temperature, gpu_uuid
        ))

    print(tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory",
                                   "temperature", "uuid")))

