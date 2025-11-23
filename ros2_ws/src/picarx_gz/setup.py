from setuptools import setup
package_name = 'picarx_gz'
setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'picarx_gz/bridge.launch.py']),
        ('share/' + package_name + '/worlds', ['worlds/factory.world']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Gazebo Harmonic world and launcher for PiCar-X',
    entry_points={'console_scripts': ['launch_world = picarx_gz.launch_world:main']},
)
