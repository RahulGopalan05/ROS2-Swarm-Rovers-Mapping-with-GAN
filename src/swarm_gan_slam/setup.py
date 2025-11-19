from setuptools import setup
import os
from glob import glob

package_name = 'swarm_gan_slam'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'models'), glob('models/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Swarm rovers mapping with GAN',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rover_controller = swarm_gan_slam.rover_controller:main',
            'map_merger = swarm_gan_slam.map_merger:main',
            'dataset_generator = swarm_gan_slam.dataset_generator:main',
            'gan_processor = swarm_gan_slam.gan_processor:main',
        ],
    },
)

