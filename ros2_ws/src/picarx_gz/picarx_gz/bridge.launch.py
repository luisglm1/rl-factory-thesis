from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/picarx/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                '/picarx/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/picarx/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/picarx/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'
            ],
            output='screen'
        )
    ])
