import numpy as np

from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.py.model import (PyLoihiProcessModel,
                                            PyAsyncProcessModel,
                                            PyLoihiModelToPyAsyncModel)
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.ports.ports import Var
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol


class SyncProc(AbstractProcess):
    def __init__(self, shape, var1: np.ndarray, var2: np.ndarray):
        super().__init__(shape=shape, var1=var1, var2=var2)

        self.in_port = InPort(shape=shape)
        self.out_port = OutPort(shape=shape)

        self.var1 = Var(shape=shape, init=var1)
        self.var2 = Var(shape=shape, init=var2)
        self.result = Var(shape=shape, init=0)


class AsyncProc(AbstractProcess):
    def __init__(self, shape, var1: np.ndarray, var2: np.ndarray):
        super().__init__(shape=shape, var1=var1, var2=var2)

        self.in_port = InPort(shape=shape)
        self.out_port = OutPort(shape=shape)

        self.var1 = Var(shape=shape, init=var1)
        self.var2 = Var(shape=shape, init=var2)

        self.num_iters = Var(shape=(1,), init=100)


@implements(proc=SyncProc, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class SyncProcModel(PyLoihiProcessModel):

    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    var1 = LavaPyType(np.ndarray, np.ndarray)
    var2 = LavaPyType(np.ndarray, np.ndarray)
    result = LavaPyType(np.ndarray, np.ndarray)

    def post_guard(self):
        return self.time_step % 10 == 0

    def run_post_mgmt(self):
        print(self.time_step)
        inp = self.in_port.recv()
        print(f"Input received at SyncProc: {inp}")
        self.result = self.result + self.var1 + self.var2 + inp
        self.out_port.send(self.result)

    def run_spk(self):
        print(f"Time step: {self.time_step}")
        self.result = self.var1 + self.var2


@implements(proc=AsyncProc, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class AsyncProcModelAsSync(PyLoihiProcessModel):

    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    var1 = LavaPyType(np.ndarray, np.ndarray)
    var2 = LavaPyType(np.ndarray, np.ndarray)
    num_iters = LavaPyType(int, int)

    def run_spk(self):

        self.out_port.send(self.var1)
        input = self.in_port.recv()
        print(input)

        self.var1 = self.var2 / input


@implements(proc=AsyncProc, protocol=AsyncProtocol)
@requires(CPU)
@tag('floating_pt')
class AsyncProcModelProper(PyAsyncProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    var1 = LavaPyType(np.ndarray, np.ndarray)
    var2 = LavaPyType(np.ndarray, np.ndarray)
    num_iters = LavaPyType(int, int)

    def __init__(self, proc_params):
        PyAsyncProcessModel.__init__(self, proc_params)
        self.time_step = 1

    def run_async(self):
        while True:
            self.time_step += 1
            self.out_port.send(self.var1)
            input = self.in_port.recv()
            print(f"From AsyncModelProper {input}")
            self.var1 = self.var2 / input
            if self.time_step == 101:
                break
        self._req_pause = True


AsyncProcModelFromLoihi = PyLoihiModelToPyAsyncModel(AsyncProcModelAsSync)


def main():

    shape = (5,)
    ones = np.ones(shape, dtype=float)
    zeros = np.zeros(shape, dtype=float)

    sync_proc = SyncProc(shape=shape,
                         var1=0.5 * ones,
                         var2=zeros)

    async_proc = AsyncProc(shape=shape,
                           var1=1.5 * ones,
                           var2=5 * ones)

    async_proc.out_port.connect(sync_proc.in_port)
    sync_proc.out_port.connect(async_proc.in_port)

    run_cond = RunSteps(num_steps=100)
    # run_cond = RunContinuous()
    pdict = {SyncProc: SyncProcModel,
             AsyncProc: AsyncProcModelProper}
    run_config = Loihi1SimCfg(
        exception_proc_model_map=pdict)

    sync_proc.run(condition=run_cond, run_cfg=run_config)
    sync_proc.stop()


if __name__ == '__main__':
    print("Running main")
    main()
