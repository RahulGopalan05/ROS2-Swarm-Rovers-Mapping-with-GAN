import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    pkg_name = 'swarm_gan_slam'
    pkg_dir = get_package_share_directory(pkg_name)
    
    world_file = os.path.join(pkg_dir, 'worlds', 'swarm_world.world')
    model_file = os.path.join(pkg_dir, 'models', 'rover.sdf')
    
    # Get gazebo_ros package directory
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    
    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'verbose': 'true',
            'world': world_file
        }.items()
    )

    # --- 🔹 Static Transform Publisher (Permanent map→odom transform) ---
    static_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )
    # Static transform for lidar
    static_base_to_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_base_to_lidar',
        arguments=['0', '0', '0.2', '0', '0', '0', 'base_link', 'lidar_link'],
        output='screen'
    )
    # --------------------------------------------------------------

    # Spawn rovers
    spawn_rover1 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'rover1',
            '-file', model_file,
            '-x', '-3.5', '-y', '-3.5', '-z', '0.15',
            '-robot_namespace', 'rover1'
        ],
        output='screen'
    )

    spawn_rover2 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'rover2',
            '-file', model_file,
            '-x', '3.5', '-y', '-3.5', '-z', '0.15',
            '-robot_namespace', 'rover2'
        ],
        output='screen'
    )

    spawn_rover3 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'rover3',
            '-file', model_file,
            '-x', '0.0', '-y', '3.5', '-z', '0.15',
            '-robot_namespace', 'rover3'
        ],
        output='screen'
    )

    # SLAM nodes for each rover
    slam_params = {
        'use_sim_time': True,
        'base_frame': 'base_link',
        'odom_frame': 'odom',
        'map_frame': 'map',
        'transform_timeout': 1.0,
    }

    slam1 = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        namespace='rover1',
        parameters=[{**slam_params, 'scan_topic': '/rover1/scan'}],
        remappings=[('/map', '/rover1/map'), ('/map_metadata', '/rover1/map_metadata')],
        output='screen'
    )

    slam2 = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        namespace='rover2',
        parameters=[{**slam_params, 'scan_topic': '/rover2/scan'}],
        remappings=[('/map', '/rover2/map'), ('/map_metadata', '/rover2/map_metadata')],
        output='screen'
    )

    slam3 = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        namespace='rover3',
        parameters=[{**slam_params, 'scan_topic': '/rover3/scan'}],
        remappings=[('/map', '/rover3/map'), ('/map_metadata', '/rover3/map_metadata')],
        output='screen'
    )

    # Controllers
    controller1 = Node(
        package='swarm_gan_slam',
        executable='rover_controller',
        name='rover_controller',
        namespace='rover1',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    controller2 = Node(
        package='swarm_gan_slam',
        executable='rover_controller',
        name='rover_controller',
        namespace='rover2',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    controller3 = Node(
        package='swarm_gan_slam',
        executable='rover_controller',
        name='rover_controller',
        namespace='rover3',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # Map merger, GAN processor, and dataset generator
    map_merger = Node(
        package='swarm_gan_slam',
        executable='map_merger',
        name='map_merger',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    gan_processor = Node(
        package='swarm_gan_slam',
        executable='gan_processor',
        name='gan_processor',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    dataset_generator = Node(
        package='swarm_gan_slam',
        executable='dataset_generator',
        name='dataset_generator',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # Return launch description with static_map_to_odom added early
    return LaunchDescription([
        gazebo,
        static_map_to_odom, 
        static_base_to_lidar, # 👈 Ensures map→odom exists before SLAM starts
        TimerAction(period=5.0, actions=[spawn_rover1]),
        TimerAction(period=6.0, actions=[spawn_rover2]),
        TimerAction(period=7.0, actions=[spawn_rover3]),
        TimerAction(period=10.0, actions=[slam1, slam2, slam3]),
        TimerAction(period=13.0, actions=[controller1, controller2, controller3]),
        TimerAction(period=15.0, actions=[map_merger]),
        TimerAction(period=18.0, actions=[dataset_generator, gan_processor]),
    ])

