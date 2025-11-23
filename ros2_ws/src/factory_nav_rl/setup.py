from setuptools import setup
package_name = 'factory_nav_rl'
setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='RL env + SB3 training (ROS2/Gazebo bridged)',
    entry_points={'console_scripts': [
        'train_ppo = factory_nav_rl.train_ppo:main',
        'train_curriculum = factory_nav_rl.train_curriculum:main',
    ]},
)
