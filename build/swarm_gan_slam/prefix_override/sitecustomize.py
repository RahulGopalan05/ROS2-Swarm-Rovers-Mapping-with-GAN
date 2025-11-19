import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/rahul/swarm_gan_slam_ws/install/swarm_gan_slam'
