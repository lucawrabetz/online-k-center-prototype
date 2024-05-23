from models import OfflineMIP, OnlineCAlgorithm
from util import FLFullInstance, INSTANCE_SHAPES

def main():
    instance = FLFullInstance(INSTANCE_SHAPES["test"])
    instance.set_random()
    instance.print()
    mip = OfflineMIP()
    mip.configure_solver(instance)
    mip.write_model("test.lp")
    # mip.solve()
    ca = OnlineCAlgorithm()
    ca.configure_solver(instance)
    ca.solve()

if __name__ == '__main__':
    main()
