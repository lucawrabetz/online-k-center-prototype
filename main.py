from models import OfflineMIP
from util import FLFullInstance, INSTANCE_SHAPES

def main():
    instance = FLFullInstance(INSTANCE_SHAPES["test"])
    instance.set_x_random()
    instance.print()
    mip = OfflineMIP()
    mip.configure_model(instance)
    mip.write_model("test.lp")
    mip.solve()

if __name__ == '__main__':
    main()
