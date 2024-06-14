from nvitop import Device
from dataclasses import dataclass, field
from typing import Optional, List

import uuid
import datetime

def get_uuid():
    return uuid.uuid4()
    
def get_date():
    return datetime.datetime.now()
    
@dataclass
class MemoryUse:
    total: float
    used: float
    free: float
     
    unit: str = 'MiB'
    
@dataclass
class GpuResourceUse:
    created: datetime.datetime = field(init=False, default_factory=get_date)
    updated: datetime.datetime = field(init=False, default_factory=get_date)
    
    name: str
    fan_speed: Optional[int] = None
    temperature: Optional[int] = None
    utilization: Optional[int] = None
    used: Optional[MemoryUse] = None
    id: uuid.UUID = field(init=True, default_factory=get_uuid)
    
    def update(self, fan_speed: int, temperature: int, utilization: int, total_mem: float, used_mem: float, free_mem: float):
        self.fan_speed = fan_speed
        self.temperature = temperature
        self.utilization = utilization
        self.used = MemoryUse(total_mem, used_mem, free_mem)
    
class GpuHandler:
    def __init__(self, processes: List[int] = None):
        self.processes = processes
        self.gpus = {}
        
        self.get_devices()
        
    def get_devices(self):
        if self.processes:
            self.devices = {key: val for key, val in Device.all() if key in self.processes}
        else:
            self.devices = Device.all()
            
        for device in self.devices:
            _id = get_uuid()
            
            self.gpus[_id] = [device, GpuResourceUse(device.name(), id=_id)]
            
    def pull(self):
        res = {}
        
        for _id, vals in self.gpus.items():
            data = vals[0]
            
            self.gpus[_id][1].update(data.fan_speed(), data.temperature(), data.gpu_utilization(), data.memory_total_human(), data.memory_used_human(), data.memory_free_human())
            
            res[_id] = self.gpus[_id][1]
            
        return res
        
if __name__ == "__main__":
    gpus = GpuHandler()
    
    data = gpus.pull()
    
    print(data, data.values())
