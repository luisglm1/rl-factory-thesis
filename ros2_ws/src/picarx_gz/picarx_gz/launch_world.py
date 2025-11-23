import os, time
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def main():
    world = os.path.join(get_package_share_directory('picarx_gz'), 'worlds', 'factory.world')
    print(f"Launching Gazebo with world: {world}")
    # start Gazebo sim (background)
    os.system(f"gz sim {world} &")
    time.sleep(5)
    print("Launching the bridge")
    # start the bridge
    bridge_launch_file = os.path.join(get_package_share_directory('picarx_gz'), 'bridge.launch.py')
    os.system(f"ros2 launch picarx_gz bridge.launch.py")

if __name__ == '__main__':
    main()