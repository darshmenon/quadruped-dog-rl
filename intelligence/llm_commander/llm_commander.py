"""
LLM Commander — natural language to robot commands using Claude API.

Translates human instructions into structured velocity/gait commands
published to ROS2 topics.

Examples:
    "walk forward slowly"       -> linear.x = 0.3
    "turn left"                 -> angular.z = 0.8
    "trot to the door"          -> gait=trot, linear.x = 0.8
    "stop"                      -> all zero
    "sit down"                  -> special pose command

Requirements:
    pip install anthropic rclpy

Usage:
    python3 intelligence/llm_commander/llm_commander.py
"""

import json
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import anthropic


SYSTEM_PROMPT = """
You are a quadruped robot controller. Convert natural language instructions
into robot commands as JSON. Output ONLY valid JSON, no explanation.

JSON format:
{
  "linear_x": <float, forward/back m/s, range -1.5 to 1.5>,
  "linear_y": <float, strafe left/right m/s, range -0.5 to 0.5>,
  "angular_z": <float, turn rad/s, range -1.5 to 1.5>,
  "gait": <"stand" | "walk" | "trot" | "canter" | "bound">,
  "action": <"move" | "stop" | "sit" | "stand">,
  "description": <one-line summary of command>
}

Rules:
- "stop", "halt", "freeze" -> all zeros, action: stop
- "sit", "lie down" -> action: sit
- "stand up", "get up" -> action: stand, gait: stand
- Speeds: slowly=0.3, normal=0.6, fast=1.2
- Turning: slightly=0.3, turn=0.8, sharp=1.5
"""


class LLMCommander(Node):
    def __init__(self):
        super().__init__('llm_commander')

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            self.get_logger().error("ANTHROPIC_API_KEY not set.")
            raise RuntimeError("Set ANTHROPIC_API_KEY environment variable.")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.action_pub = self.create_publisher(String, '/robot_action', 10)
        self.input_sub = self.create_subscription(
            String, '/natural_language_cmd', self.command_cb, 10
        )
        self.get_logger().info('LLM Commander ready. Publish to /natural_language_cmd')

    def command_cb(self, msg: String):
        instruction = msg.data
        self.get_logger().info(f'Received: "{instruction}"')

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": instruction}]
            )
            raw = response.content[0].text.strip()
            cmd_data = json.loads(raw)
        except Exception as e:
            self.get_logger().error(f'LLM parse error: {e}')
            return

        self.get_logger().info(f'Command: {cmd_data.get("description", "")}')

        twist = Twist()
        twist.linear.x = float(cmd_data.get("linear_x", 0.0))
        twist.linear.y = float(cmd_data.get("linear_y", 0.0))
        twist.angular.z = float(cmd_data.get("angular_z", 0.0))
        self.cmd_pub.publish(twist)

        action_msg = String()
        action_msg.data = cmd_data.get("action", "move")
        self.action_pub.publish(action_msg)


def main():
    rclpy.init()
    node = LLMCommander()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
