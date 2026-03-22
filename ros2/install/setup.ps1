# generated from colcon_powershell/shell/template/prefix_chain.ps1.em

# This script extends the environment with the environment of other prefix
# paths which were sourced when this file was generated as well as all packages
# contained in this prefix path.

# function to source another script with conditional trace output
# first argument: the path of the script
function _colcon_prefix_chain_powershell_source_script {
  param (
    $_colcon_prefix_chain_powershell_source_script_param
  )
  # source script with conditional trace output
  if (Test-Path $_colcon_prefix_chain_powershell_source_script_param) {
    if ($env:COLCON_TRACE) {
      echo ". '$_colcon_prefix_chain_powershell_source_script_param'"
    }
    . "$_colcon_prefix_chain_powershell_source_script_param"
  } else {
    Write-Error "not found: '$_colcon_prefix_chain_powershell_source_script_param'"
  }
}

# source chained prefixes
_colcon_prefix_chain_powershell_source_script "/opt/ros/humble\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/asimov/fleet_management_beetle-arm/mobile_robots/navigation/install\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/asimov/chess-demo/install\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/asimov/pickplace-rl-mobile/src/pickplace_rl_mobile/install\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/asimov/pickplace-rl-mobile/install\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/asimov/robot_arm_workspace/install\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/asimov/install\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/asimov/fleet_management_beetle-arm/install\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/asimov/animatronics_head_ros2/install\local_setup.ps1"
_colcon_prefix_chain_powershell_source_script "/home/asimov/facial-animatronic-ros2/install\local_setup.ps1"

# source this prefix
$env:COLCON_CURRENT_PREFIX=(Split-Path $PSCommandPath -Parent)
_colcon_prefix_chain_powershell_source_script "$env:COLCON_CURRENT_PREFIX\local_setup.ps1"
