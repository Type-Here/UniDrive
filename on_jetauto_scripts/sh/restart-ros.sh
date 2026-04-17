#!/bin/bash
# Avvia RTAB-Map SLAM (RGB-D + LIDAR A1)
gnome-terminal \
--tab -e "zsh -c 'source $HOME/jetauto_ws/.zshrc;sudo systemctl stop start_app_node;killall -9 rosmaster;roslaunch jetauto_slam slam.launch robot_name:=/ master_name:=/ slam_methods:=rtabmap'" \
--tab -e "zsh -c 'source $HOME/jetauto_ws/.zshrc;sleep 30;roscd jetauto_slam/rviz;rviz -d rtabmap.rviz'" \
--tab -e "zsh -c 'source $HOME/jetauto_ws/.zshrc;sleep 30;roslaunch jetauto_peripherals teleop_key_control.launch robot_name:=/'"