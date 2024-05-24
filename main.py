from solvers import OfflineMIP, OnlineCVCTAlgorithm
from problem import FLFullInstance, INSTANCE_SHAPES

def main():
    instance = FLFullInstance(INSTANCE_SHAPES["small"])
    instance.set_random()
    instance.print()
    mip = OfflineMIP()
    mip.configure_solver(instance)
    print("SOLVING OFFLINE MIP...")
    print()
    mip.solve()
    print()
    print("SOLVING ONLINE CVTCA...")
    algo = OnlineCVCTAlgorithm()
    algo.configure_solver(instance)
    algo.solve()

if __name__ == '__main__':
    main()
